"""
Condition-preset engine (track / car / surface-friction switching).

Reimplemented per docs/condition_module_spec.md — the original implementation
lived in the private training repo (/home/desc/gym_torcs/snn_torcs/condition.py)
and was NOT part of the two handoff packages. It is the engine behind
`scripts/eval_condition.py` and the robustness matrix
`scripts/run_7_condition_h20_matrix.sh`, switching TORCS conditions defined in
conditions/conditions.yaml:

    baseline / wet / icy / wet-all / alt-car / alt-track

Interface (all call sites in scripts/eval_condition.py):

    condition = cond_mod.load_condition(name)       # preset from the yaml
    before    = cond_mod.snapshot_stock()           # sha256 of stock files
    condition.apply()                               # switch TORCS configs
    cond_mod.install_signal_handlers(condition, before)
    ...
    condition.restore()                             # put stock back
    cond_mod.assert_stock_unchanged(before)         # pollution gate

Semantics — the spec allows the original "cwd symlink-farm" (zero stock writes
during the run) to degrade to the equivalent backup/patch/restore scheme, which
is what this implementation does (same semantics as run_mode.py's
backup_configs/restore_configs for practice.xml/screen.xml):

    apply()    backs up every stock file it will touch (suffix .condbak.<pid>),
               then patches it in place
    restore()  copies the backups back — stock ends byte-identical
    gate       snapshot_stock() / assert_stock_unchanged() compare sha256 of
               every stock path the module (or run_mode) may touch

The three TORCS config edits (scoped, per conditions.yaml header):
  1. track select  — ~/.torcs/config/raceman/practice.xml
        <section name="Tracks"><section name="1"> attstr name/category
  2. car select    — scr_server.xml <attstr name="car name"> (global, all
        robot index sections — documented safe)
  3. grip select   — track XML <section name="<surface>"> attnum friction
        (scoped to that surface section only)
"""
import glob
import hashlib
import os
import re
import shutil
import signal

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
CONDITIONS_YAML = os.path.join(_REPO_ROOT, "conditions", "conditions.yaml")

# Reuse run_mode's notion of the user config dir so the two patchers can never
# disagree about paths.
from snn_torcs.run_mode import PRACTICE_XML, SCREEN_XML  # noqa: E402

# Some deployments (this docker snapshot included) resolve the Practice race
# from the *system* raceman file instead of ~/.torcs — empirically the race
# track follows the system copy (probed 2026-10: user-only patch is a no-op,
# system+user patch switches the track). Patch BOTH to be safe on either
# layout; the pollution gate snapshots both.
_PRACTICE_SYS_CANDIDATES = [
    os.path.join(root, "config", "raceman", "practice.xml")
    for root in ("/usr/local/share/games/torcs", "/usr/share/games/torcs")
]

_BACKUP_SUFFIX = ".condbak.{0}".format(os.getpid())

# scr_server.xml location varies with the TORCS variant / install prefix —
# search the known candidates and fail loudly if none exists.
_SCR_SERVER_CANDIDATES = [
    os.path.expanduser("~/.torcs/drivers/scr_server/scr_server.xml"),
    "/usr/local/share/games/torcs/drivers/scr_server/scr_server.xml",
    "/usr/share/games/torcs/drivers/scr_server/scr_server.xml",
]

_TRACK_DATA_ROOT_CANDIDATES = [
    "/usr/local/share/games/torcs",
    "/usr/share/games/torcs",
]


# ------------------------------------------------------------------ #
# path resolution
# ------------------------------------------------------------------ #
def find_practice_xmls():
    """Every practice.xml that may select the race track: the user copy
    (~/.torcs, per the original design) plus the system copy (read by some
    deployments — see _PRACTICE_SYS_CANDIDATES note). Existing files only."""
    paths = [PRACTICE_XML]
    for cand in _PRACTICE_SYS_CANDIDATES:
        if os.path.isfile(cand) and cand not in paths:
            paths.append(cand)
    return paths


def find_scr_server_xml():
    for cand in _SCR_SERVER_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    hits = glob.glob("/usr/*/share/games/torcs/drivers/scr_server/scr_server.xml")
    if hits:
        return sorted(hits)[0]
    raise RuntimeError(
        "scr_server.xml not found in any of: {0}".format(_SCR_SERVER_CANDIDATES))


def find_track_xml(track, category):
    for root in _TRACK_DATA_ROOT_CANDIDATES:
        cand = os.path.join(root, "tracks", category, track, track + ".xml")
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        "track XML for {0}/{1} not found under {2}".format(
            category, track, _TRACK_DATA_ROOT_CANDIDATES))


def _all_track_xmls_from_yaml(presets):
    """Every track XML any preset may touch — snapshot the whole set so the
    pollution gate also catches accidental writes to the *other* track."""
    paths = {}
    for name, spec in presets.items():
        try:
            paths[name] = find_track_xml(spec["track"], spec["category"])
        except RuntimeError:
            paths[name] = None
    return paths


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ #
# scoped XML patchers (regex, whitespace/attr preserving — same style as
# run_mode._patch_xml_attr)
# ------------------------------------------------------------------ #
def _patch_practice_track(content, track, category):
    """Scoped edit of <section name="Tracks"><section name="1"> only."""
    i = content.find('<section name="Tracks">')
    if i < 0:
        raise RuntimeError("practice.xml: <section name=\"Tracks\"> not found")
    # first close after the Tracks opening tag closes sub-section "1"
    j = content.find("</section>", i)
    if j < 0:
        raise RuntimeError("practice.xml: Tracks section not terminated")
    span = content[i:j]
    span, n_name = re.subn(
        r'(<attstr\s+name="name"\s+val=")([^"]*)(")',
        lambda m: m.group(1) + track + m.group(3), span, count=1)
    span, n_cat = re.subn(
        r'(<attstr\s+name="category"\s+val=")([^"]*)(")',
        lambda m: m.group(1) + category + m.group(3), span, count=1)
    if n_name != 1 or n_cat != 1:
        raise RuntimeError(
            "practice.xml: Tracks/1 name(%d)/category(%d) not uniquely found"
            % (n_name, n_cat))
    return content[:i] + span + content[j:]


def _patch_scr_server_car(content, car):
    """Global edit of <attstr name="car name"> (all robot index sections)."""
    new, n = re.subn(
        r'(<attstr\s+name="car name"\s+val=")([^"]*)(")',
        lambda m: m.group(1) + car + m.group(3), content)
    if n == 0:
        raise RuntimeError('scr_server.xml: no <attstr name="car name"> found')
    return new


def _patch_track_friction(content, surface, value):
    """Scoped edit: the friction attnum *inside* <section name="<surface>">.

    The tempered (?:(?!</section>).)*? guard keeps the match from leaking past
    the section's closing tag (surface sections do not nest).
    """
    pat = (r'(<section name="' + re.escape(surface) + r'">'
           r'(?:(?!</section>).)*?'
           r'<attnum\s+name="friction"\s+val=")([^"]*)(")')
    new, n = re.subn(
        pat, lambda m: m.group(1) + value + m.group(3),
        content, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(
            'track XML: surface "%s" friction not uniquely found' % surface)
    return new


def _read_friction(content, surface):
    m = re.search(
        r'<section name="' + re.escape(surface) + r'">'
        r'(?:(?!</section>).)*?<attnum\s+name="friction"\s+val="([^"]*)"',
        content, flags=re.DOTALL)
    return m.group(1) if m else None


# ------------------------------------------------------------------ #
# Condition object
# ------------------------------------------------------------------ #
class Condition(object):
    def __init__(self, name, spec, scr_server_xml):
        self.name = name
        self.spec = spec
        self.scr_server_xml = scr_server_xml
        self._backups = []   # [(target_path, backup_path)] in apply order

    # -- internal helpers ------------------------------------------- #
    def _track_xml(self):
        return find_track_xml(self.spec["track"], self.spec["category"])

    def _backup(self, path):
        for tgt, _ in self._backups:
            if tgt == path:
                return
        bak = path + _BACKUP_SUFFIX
        if os.path.exists(bak):
            raise RuntimeError("stale backup exists: %s" % bak)
        shutil.copy2(path, bak)
        self._backups.append((path, bak))

    def _rewrite(self, path, patcher, what):
        with open(path, "r") as f:
            content = f.read()
        new = patcher(content)
        with open(path, "w") as f:
            f.write(new)
        return what

    # -- public API --------------------------------------------------- #
    def apply(self):
        """Back up then patch the three TORCS config files for this preset."""
        track = self.spec["track"]
        category = self.spec["category"]
        car = self.spec["car"]
        frictions = self.spec.get("friction") or []

        practice_paths = find_practice_xmls()
        scr = self.scr_server_xml
        txml = self._track_xml()
        for p in practice_paths + [scr, txml]:
            if not os.path.isfile(p):
                raise RuntimeError("condition target missing: %s" % p)

        # backup everything FIRST (no half-applied state on patch failure)
        for p in practice_paths + [scr, txml]:
            self._backup(p)

        # 1. track select (scoped to Tracks/1) — user AND system copies
        for p in practice_paths:
            self._rewrite(
                p, lambda c: _patch_practice_track(c, track, category),
                "practice.xml track=%s/%s" % (track, category))

        # 2. car select (global, safe)
        self._rewrite(
            scr, lambda c: _patch_scr_server_car(c, car),
            "scr_server.xml car=%s" % car)

        # 3. grip select (scoped per surface)
        def _fric(c):
            old_new = []
            for e in frictions:
                old = _read_friction(c, e["surface"])
                c = _patch_track_friction(c, e["surface"], str(e["value"]))
                old_new.append((e["surface"], old, str(e["value"])))
            self._friction_log = old_new
            return c

        self._rewrite(txml, _fric, "track xml friction")

        print("[condition:%s] patched practice.xml(%s/%s) scr_server.xml(%s) "
              "%s(friction %s)" % (
                  self.name, track, category, car, os.path.basename(txml),
                  getattr(self, "_friction_log", [])))

    def restore(self):
        """Copy the pre-apply backups back (reverse order)."""
        errors = []
        for path, bak in reversed(self._backups):
            try:
                if os.path.isfile(bak):
                    shutil.copy2(bak, path)
                    os.remove(bak)
            except Exception as e:            # pragma: no cover
                errors.append((path, str(e)))
        self._backups = []
        if errors:
            raise RuntimeError("restore failed: %r" % (errors,))


# ------------------------------------------------------------------ #
# module-level API (mirrors the original handoff interface)
# ------------------------------------------------------------------ #
def load_condition(name):
    """Return a condition object with .spec / .apply() / .restore()."""
    with open(CONDITIONS_YAML) as f:
        presets = yaml.safe_load(f)
    if name not in presets:
        raise ValueError(
            "unknown condition preset %r (have: %s)" % (name, sorted(presets)))
    spec = presets[name]
    for key in ("track", "category", "car"):
        if key not in spec:
            raise ValueError("preset %r missing key %r" % (name, key))
    spec.setdefault("friction", [])
    scr = find_scr_server_xml()
    # fail fast if the preset's track is not installed
    find_track_xml(spec["track"], spec["category"])
    return Condition(name, spec, scr)


def _stock_paths():
    """All stock paths this module (or run_mode during a condition run) may
    touch: practice.xml (user + system), screen.xml, scr_server.xml and every
    track XML referenced by any preset."""
    paths = [PRACTICE_XML, SCREEN_XML, find_scr_server_xml()]
    paths.extend(p for p in find_practice_xmls() if p not in paths)
    with open(CONDITIONS_YAML) as f:
        presets = yaml.safe_load(f)
    for p in _all_track_xmls_from_yaml(presets).values():
        if p and p not in paths:
            paths.append(p)
    return paths


def snapshot_stock():
    """Snapshot stock TORCS files (sha256) before a condition run."""
    files = {}
    for p in _stock_paths():
        if os.path.isfile(p):
            files[p] = _sha256(p)
        else:
            files[p] = None          # missing now -> must still be missing later
    return {"files": files}


def assert_stock_unchanged(before):
    """Pollution gate: raise if any stock file drifted."""
    before_files = before["files"]
    drifted = []
    for p, old in before_files.items():
        if not os.path.isfile(p):
            now = None
        else:
            now = _sha256(p)
        if now != old:
            drifted.append((p, old, now))
    if drifted:
        raise RuntimeError(
            "pollution gate FAILED — stock files drifted: %r" % (drifted,))
    return True


def install_signal_handlers(condition, before):
    """Restore stock + run the pollution gate on SIGINT/SIGTERM."""
    def _handler(signum, frame):
        try:
            condition.restore()
        finally:
            try:
                assert_stock_unchanged(before)
                print("[condition] pollution gate PASSED (signal %d) — "
                      "stock unchanged" % signum)
            except Exception as e:
                print("[condition] pollution gate FAILED (signal %d): %s"
                      % (signum, e))
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

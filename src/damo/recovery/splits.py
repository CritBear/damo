import hashlib
import json
from pathlib import Path, PurePosixPath
from .base import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_cache(root):
    cache = json.loads((Path(root) / 'cache_manifest.json').read_text(encoding='utf-8'))
    if cache.get('format') != 'damo-body22-cache-v1' or not cache.get('complete'):
        raise ValueError('A complete native body22 cache is required')
    return cache


def inventory(cache):
    mapping = {}
    for key, motion in cache.get('motions', {}).items():
        for path in motion['clips']:
            if path in mapping:
                raise ValueError('A clip belongs to multiple original motions')
            mapping[path] = (key, motion)
    for path, item in cache['files'].items():
        name = PurePosixPath(path).parent.name
        if path not in mapping:
            key = name + '__' + item['motion']
            mapping[path] = (key, {**item, 'dataset': name})
        if mapping[path][1]['dataset'] != name:
            raise ValueError('Inconsistent motion dataset')
    if set(mapping) != set(cache['files']):
        raise ValueError('Motion index and cache files differ')
    return mapping


def create_split(data_root, output, *, val_fraction=.2, seed=2024):
    root = Path(data_root).resolve()
    cache = read_cache(root)
    mapping = inventory(cache)
    names = sorted({m['dataset'] for _, m in mapping.values()})
    if not 0 < val_fraction < 1:
        raise ValueError('Validation fraction must lie between zero and one')
    roles, counts = {}, {}
    for name in names:
        keys = {k for k, m in mapping.values() if m['dataset'] == name}
        if len(keys) < 2:
            raise ValueError(f'{name}: need at least two original motions')
        ordered = sorted(keys, key=lambda key: digest([seed, 'all7_whole_motion', key]))
        n = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
        roles.update({key: 'val' if i < n else 'train' for i, key in enumerate(ordered)})
        counts[name] = dict(train=len(keys) - n, val=n)
    entries, partitions, source_roles = {}, {'train': [], 'val': [], 'test': []}, {}
    for path, item in sorted(cache['files'].items()):
        key, motion = mapping[path]
        role = roles[key]
        source = motion.get('source_sha256') or item.get('source_sha256')
        if source and source_roles.setdefault(source, role) != role:
            raise ValueError('Duplicate source content crosses partitions; deduplicate the raw recordings first')
        partitions[role].append(path)
        entries[path] = dict(group=key, sha256=item['sha256'], frames=item['frames'],
                             source_group=motion['dataset'] + '/source:' + str(motion.get('subject', key)))
    manifest = dict(format='damo-whole-motion-split-v1', seed=seed, val_fraction=val_fraction,
                    train_names=names, test_names=[], common_sha256=cache['common_sha256'],
                    cache_sha256=sha256(root / 'cache_manifest.json'), motion_roles=roles,
                    partitions=partitions, entries=entries, counts=counts,
                    grouping='Within each dataset, whole original motions; source identities may overlap; no independent test')
    manifest['id'] = digest(manifest)
    output = Path(output)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise FileExistsError('A different split already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def resolve_split(cfg):
    root = Path(cfg['data_root']).resolve()
    data = cfg['dataset']
    path = Path(data['split_manifest'])
    manifest = json.loads((path if path.is_absolute() else root / path).read_text(encoding='utf-8'))
    identity = manifest.pop('id')
    if manifest.get('format') != 'damo-whole-motion-split-v1' or digest(manifest) != identity:
        raise ValueError('Split identity mismatch')
    for key in ('train_names', 'val_names'):
        if set(data[key]) != set(manifest['train_names']):
            raise ValueError('Configured datasets differ from split')
    if data.get('test_names') or manifest['partitions']['test']:
        raise ValueError('This protocol has train/validation only')
    cache = read_cache(root)
    if sha256(root / 'cache_manifest.json') != manifest['cache_sha256']:
        raise ValueError('Cache changed after splitting')
    if sha256(cfg['common']) != manifest['common_sha256'] or cache['common_sha256'] != manifest['common_sha256']:
        raise ValueError('Common base differs from split')
    mapping = inventory(cache)
    result, seen, groups, sources = {}, set(), set(), {}
    for role in ('train', 'val', 'test'):
        paths = manifest['partitions'][role]
        keys = {mapping[p][0] for p in paths}
        if (role != 'test' and not paths) or len(set(paths)) != len(paths) or seen.intersection(paths) or groups.intersection(keys):
            raise ValueError('Empty, duplicated or leaking split')
        seen.update(paths)
        groups.update(keys)
        result[role] = []
        for rel in paths:
            path = (root / rel).resolve()
            key, motion = mapping[rel]
            expected = manifest['entries'][rel]
            if root not in path.parents or not path.is_file():
                raise ValueError('Missing or unsafe cache path')
            if expected['group'] != key or manifest['motion_roles'][key] != role:
                raise ValueError('Original-motion assignment mismatch')
            source = motion.get('source_sha256') or cache['files'][rel].get('source_sha256')
            if source and sources.setdefault(source, role) != role:
                raise ValueError('Duplicated source crosses train and validation')
            if sha256(path) != expected['sha256'] or cache['files'][rel]['sha256'] != expected['sha256']:
                raise ValueError(f'Cache file changed: {rel}')
            result[role].append(path)
    if seen != set(cache['files']):
        raise ValueError('Every prepared clip must be assigned exactly once')
    for role in ('train', 'val'):
        if {p.parent.name for p in result[role]} != set(manifest['train_names']):
            raise ValueError('Every dataset must occur in both partitions')
    return result, identity

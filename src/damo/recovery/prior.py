import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from .base import sha256
from .data import load_common, load_clip
from .splits import resolve_split, read_cache, inventory
from .anchor import CONFIG, SYMMETRY, quantile


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def build_prior(cfg, output):
    parts, identity = resolve_split(cfg)
    split = {'id': identity}
    common = load_common(cfg['common'])
    cache = read_cache(cfg['data_root'])
    mapping = inventory(cache)
    groups = {}
    for path in parts['train']:
        rel = path.relative_to(Path(cfg['data_root']).resolve()).as_posix()
        key, motion = mapping[rel]
        groups.setdefault(key, (path, motion))
    records = []
    for key, (path, motion) in sorted(groups.items()):
        joints = np.asarray(motion['bind_jgp'] if 'bind_jgp' in motion else load_clip(path)['bind_jgp'], float)
        group = motion['dataset'] + '/source:' + str(motion.get('subject', key))
        records.append((key, group, joints - joints[0]))
    if len(records) < 2:
        raise ValueError('At least two train shapes are required')
    counts=Counter(g for _,g,_ in records)
    wr=np.array([1/counts[g] for _,g,_ in records]);wr=wr/wr.sum()
    j=np.array([r[2] for r in records]);w=wr
    parents=np.asarray(common['topology']);scale=np.sqrt(np.mean(j[:,1:]**2,axis=(1,2)))
    normalized=(j[:,1:]/scale[:,None,None]).reshape(len(j),-1)
    mean=np.average(normalized,axis=0,weights=w);center=normalized-mean
    values,vectors=np.linalg.eigh((center*w[:,None]).T@center);idx=np.argsort(values)[::-1];values=values[idx];vectors=vectors[:,idx]
    if not np.isfinite(values).all() or values.sum() <= 1e-12:
        raise ValueError('Train shapes have no usable shape variation')
    count=min(CONFIG['pca_max_components'],int(np.searchsorted(np.cumsum(values)/values.sum(),CONFIG['pca_variance']))+1)
    basis=vectors[:,:count]*np.sqrt(np.maximum(values[:count],0))[None,:]
    bones=j[:,1:]-j[:,parents[1:]];length=np.linalg.norm(bones,axis=-1);directions=bones/length[...,None]
    md=np.average(directions,axis=0,weights=w);md/=np.linalg.norm(md,axis=-1)[:,None]
    angles=np.arccos(np.clip(np.sum(directions*md,axis=-1),-1,1));pairs=np.array(SYMMETRY)-1
    ratios=np.log(length[:,pairs[:,0]]/length[:,pairs[:,1]]);rm=np.average(ratios,axis=0,weights=w)
    rs=np.maximum(np.sqrt(np.average((ratios-rm)**2,axis=0,weights=w)),.03);sl=np.average(np.log(scale),weights=w)
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'shape_prior.npz',mean=mean,basis=basis,log_scale_mean=sl,
        log_scale_std=max(float(np.sqrt(np.average((np.log(scale)-sl)**2,weights=w))),.03),
        log_scale_bounds=np.log([quantile(scale,w,.01)*.9,quantile(scale,w,.99)*1.1]),
        length_low=[quantile(length[:,i],w,.01)*.9 for i in range(21)],
        length_high=[quantile(length[:,i],w,.99)*1.1 for i in range(21)],mean_direction=md,
        direction_limit=[quantile(angles[:,i],w,.99)+CONFIG['direction_allowance_rad'] for i in range(21)],
        ratio_mean=rm,ratio_std=rs,parents=parents,weights=w)
    def stats(x):
        mu=np.average(x,weights=w)
        return dict(mean=float(mu),std=float(np.sqrt(np.average((x-mu)**2,weights=w))),p01=float(quantile(x,w,.01)),median=float(quantile(x,w,.5)),p99=float(quantile(x,w,.99)))
    log=np.log(length);mu=np.average(log,axis=0,weights=w)
    np.savez(dest/'train_log_length_statistics.npz',mean=mu,covariance=((log-mu)*w[:,None]).T@(log-mu),weights=w)
    dump(dest/'summary.json',dict(prior_fit_roles=['train'],split_id=split['id'],train_bones=[dict(joint=i+1,**stats(length[:,i])) for i in range(21)]))
    dump(dest/'train_joint_pair_prior.json',dict(fit_roles=['train'],pairs=[dict(joint_a=a,joint_b=b,**stats(np.linalg.norm(j[:,a]-j[:,b],axis=1))) for a in range(22) for b in range(a+1,22)]))
    dump(dest/'manifest.json',dict(created_utc=now(),split_id=split['id'],fit_roles=['train'],train_motion_keys=[r[0] for r in records],train_groups=len(counts),synthetic_training_bodies=0,weighting='new train real shapes only, equal source group and equal motion within group',components=count,variance_explained=float(values[:count].sum()/values.sum()),common_sha256=sha256(cfg['common']),heldout_shapes_used=False))
    recipe = dict(state='frozen', name='train-only shape, anchors and marker consistency', version=1,
                  created_utc=now(), split_id=identity, shape_prior='shape_prior.npz',
                  skeleton_parameters=dict(CONFIG), ground_truth_in_fit=False,
                  pose_policy=dict(backend='hybrid', max_nfev_per_attempt=100, retry_rmse_m=1e-5),
                  prior_sha256={p.name: sha256(p) for p in sorted(dest.iterdir()) if p.is_file()},
                  source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    dump(dest/'recipe.json', recipe)
    return dict(recipe=str((dest/'recipe.json').resolve()), split_id=identity, train_motions=len(records))

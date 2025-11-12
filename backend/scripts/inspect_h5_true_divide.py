import h5py, os
root = r"d:\GitHub Desktop\horus_ai\backend\app\ml\models"
for fname in os.listdir(root):
    if fname.endswith('.h5'):
        path = os.path.join(root, fname)
        print('\nFILE:', fname)
        try:
            with h5py.File(path, 'r') as f:
                # check attrs for model_config
                if 'model_config' in f.attrs:
                    mc = f.attrs['model_config']
                    try:
                        s = mc.decode() if isinstance(mc, bytes) else str(mc)
                    except Exception:
                        s = str(mc)
                    found = 'TrueDivide' in s
                    print(' model_config contains TrueDivide?', found)
                    if found:
                        idx = s.find('TrueDivide')
                        print(' ...', s[max(0, idx-120):idx+120])
                # search recursively small attrs
                def search_group(g, path='/'):
                    for k, v in g.attrs.items():
                        try:
                            s = v.decode() if isinstance(v, bytes) else str(v)
                        except Exception:
                            s = str(v)
                        if 'TrueDivide' in s:
                            print(' ATTR', path, k, 'contains TrueDivide')
                    for name, item in g.items():
                        try:
                            if isinstance(item, h5py.Group):
                                search_group(item, path+name+'/')
                            else:
                                # dataset
                                pass
                        except Exception:
                            pass
                search_group(f)
        except Exception as e:
            print(' error opening:', e)

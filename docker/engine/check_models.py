import onnxruntime as ort

for n in ['encoder', 'decoder', 'joiner']:
    path = f'/models/ASR/parakeet-tdt-0.6b-v3_{n}.onnx'
    m = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    print(f'=== {n.upper()} ===')
    for i in m.get_inputs(): print(f'  In: {i.name} {i.shape} {i.type}')
    for o in m.get_outputs(): print(f'  Out: {o.name} {o.shape} {o.type}')
    meta = m.get_modelmeta().custom_metadata_map
    for k, v in meta.items(): print(f'  Meta: {k}={v}')
    print()

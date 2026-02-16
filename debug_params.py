from api.main import MODELS_DB
m = next(x for x in MODELS_DB['models'] if 'scout' in x['id'])
print(f"ID: {m['id']}, Params: {m['parameters']}")

import sys, os, json, numpy as np
from PIL import Image
sys.path.insert(0, ".")
def main():
    from paas import env; env.setup(device="cuda:0")
    from paas.config import PaasConfig
    from paas.pipeline import PaasPipeline
    cfg = PaasConfig.from_dict(json.load(open("config/experiments/paas4_qwen.json"))).validate()
    print("[smoke] building pipeline (tf5 single-venv):", cfg.fusion.components, flush=True)
    pipe = PaasPipeline(cfg)
    tests = [("REAL","/datasets/work/vLLM/temp/testset/real/real_id_R_10__id_R_10_0_f0.jpg"),
             ("FAKE","/datasets/work/vLLM/temp/testset/fake/deepfake/df_v2_2025W01_0.png")]
    rgb=[np.array(Image.open(p).convert("RGB")) for _,p in tests]; keys=[p for _,p in tests]
    res=pipe.predict_frames(rgb, keys=keys)
    for (lab,_),r in zip(tests,res):
        print(f"\n[{lab}] forgery_score={r.get('forgery_score')} decision={r.get('decision')}", flush=True)
        print("   components:", {k:(round(v,4) if isinstance(v,float) else v) for k,v in (r.get('components') or {}).items()})
    print("\n[smoke] OK single-venv end-to-end")
if __name__=="__main__": main()

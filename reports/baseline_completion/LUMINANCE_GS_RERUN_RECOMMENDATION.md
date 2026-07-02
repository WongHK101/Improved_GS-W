# LUMINANCE_GS_RERUN_RECOMMENDATION

Recommendation: do not launch full 30k reruns.

Bounded next step if GPT approves Luminance-GS repair: fix the wrapper/environment mismatch around `pycolmap.SceneManager` first, then rerun import + reader smoke. Do not run 30k until the current import failure is resolved.

Priority scenes: `self_3000t_Press, self_Trackmobile_4650TM_Mobile_Railcar_Mover, web_metopa_images`.

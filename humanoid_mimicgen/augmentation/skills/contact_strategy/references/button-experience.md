# Grounded button experience

Historical manual study20261024_2352 and hold validation20261025_0027 found: proximal-only pressure remained about11.72N; folding index and changing wrist pose switched actual contact to right_wrist_yaw_link (palm collision). Earlier hand formation/deeper reference produced23.31N at end but only.26s continuously>=20N. Holding identical intent1.5s longer yielded1.76s>=20N, final20.93N, with exact token+hands replay.

Native task remained false: hand-body keyword filter excludes wrist_yaw. This is a documented contact-filter omission; the study did not change native acceptance. Legacy feedback also reads native force, so palm contact can leave the integral saturating. It demonstrates a physically effective bounded pose, not a verified palm-force-feedback law. One source, one scene offset X+5mm; no generalized success claim.

API may reuse the supplied parameters when conditions match, but must label reuse and freshly execute. Test new tasks from their own source geometry, task criteria and control capabilities. The useful lesson is the evidence-guided contact-region change plus duration-only validation, not a universal offset prescription.

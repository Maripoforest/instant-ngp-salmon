import threading
from pipeline import NeRFPipeline

ppl = NeRFPipeline()
ppl.delay = True
pipeline_thread = threading.Thread(target=ppl.spinning)
pipeline_thread.start()

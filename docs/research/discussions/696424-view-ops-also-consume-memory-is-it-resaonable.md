# view ops also consume memory: is it resaonable?

- Topic ID: 696424
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696424
- Author: hengck23 (@hengck23)
- Posted: 2026-05-02T08:38:21.075459200Z
- Votes: 3
- Total messages: 2

## Body

some of the ops never create new memory in practice...  examples are 

Reshape  
Flatten  
Identity  
Squeeze  
Unsqueeze   

some really good hacking algorithms rely heavily on reshape, but are penalized. I wonder if we should consider these as "zero-memory ops" in the metric?

## Comments (2)

- **Michael D. Moffitt** (2026-05-13T16:24:31.060Z, votes: {'canUpvote': True}):
  I had meant to reply to this earlier, but here are my thoughts: yes, treating these ops as "free" for the vast majority of cases (e.g., execution or hardware acceleration) makes perfect sense, esp. since they often require neither memory nor compute.
  
  For this contest, there are a few reasons to keep their costs nonzero:
   - We're trying to emphasize the simplicity of solutions overall, and since these ops are parameter-free, the memory penalty prevents teams from going crazy with them.
   - Given the risk of missing an op here or there (which would potentially require another batch rescore), it's simply easier to treat them all the same.  I'd prefer to avoid reimplementing `onnx_tool` from scratch, so hopefully this decision prevents a lot of complicated corner cases that might otherwise cause headaches for everyone.

- **NNMax** (2026-05-02T09:01:25.967Z, votes: {'canUpvote': True}):
  I think reshape should be a zero cost op or atleast consume very low memory footprint. Official onnx documentation says that it is similar to numpy.reshape and if you look at what numpy.reshape does, it just changes the way the data is presented instead of changing the underlying data. By that logic, no additional memory should be consumed when using these ops.

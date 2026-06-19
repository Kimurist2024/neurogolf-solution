# Dinamic slices and pads

- Topic ID: 704900
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/704900
- Author: Paolo Antonuccio (@paoloantonuccio)
- Posted: 2026-06-07T08:11:10.544242300Z
- Votes: 0
- Total messages: 2

## Body

In some tasks, it is necessary to use dynamic slices and pads, such as in task014, but when I run it, I get an output like this:

Results on ARC-AGI examples: 1 pass, 3 fail
Results on ARC-GEN examples: 2 pass, 260 fail

Error: Your network performance could not be measured

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1126276753.py in <cell line: 0>()
     40 onnx.checker.check_model(model)
     41 onnx.save(model, onnx_filename)
---> 42 verify_network(model, task_num, examples)

/kaggle/input/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py in verify_network(network, task_num, examples)
    505   if memory is None or params is None:
    506     print(“Error: Your network performance could not be measured”)
--> 507   if memory < 0 or params < 0:
    508     print(“Error: Your network performance could not be measured”)
    509   elif arc_agi_wrong + arc_gen_wrong == 0:

TypeError: ‘<’ not supported between instances of ‘NoneType’ and ‘int’

What should I do?

## Comments (2)

- **Bishnu Ganguly** (2026-06-07T15:45:32.107Z, votes: {'canUpvote': True}):
  For the dynamic slicing issue (if the shape of slice is constant such as it is  in task 111 where we are doing 3*3 crop)  , this issue can be sorted by Patching dynamic Slice output shape into value_info of the model  . [detailed discussion ](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695972)

  - **Paolo Antonuccio** (2026-06-08T08:05:28.820Z, votes: {'canUpvote': True}):
    I tried spox, but the result is the same

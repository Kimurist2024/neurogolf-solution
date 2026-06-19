# any onnx expert?

- Topic ID: 694463
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694463
- Author: hengck23 (@hengck23)
- Posted: 2026-04-24T23:42:15.229082400Z
- Votes: 3
- Total messages: 1

## Body

say i select the active color channel c1,c2,c3   
then i process them in memory 3xSxS

after than i need to put back to 9xSxS in onehot ouput.

in np python, this is simpley ouput[ci] =....

but chatgpt says there is no simple way to do it! either you need to broadcast to 9x3xSxS and reduce to 9xSxS or you use scatter elements or ScatterND.

is it correct?

## Comments (1)

- **Tom** (2026-04-25T00:18:08.240Z, votes: {'canUpvote': True}):
  index map seems worked

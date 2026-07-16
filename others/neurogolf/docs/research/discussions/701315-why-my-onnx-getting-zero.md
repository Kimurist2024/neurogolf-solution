# Why my onnx getting zero?

- Topic ID: 701315
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/701315
- Author: Rustam Bazarbayev (@rustambazarbayev)
- Posted: 2026-05-18T19:13:31.251645900Z
- Votes: 0
- Total messages: 2

## Body

I submitted an ONNX file but got a result of 0. Can someone help me solve the issue?

## Comments (2)

- **Lixin73** (2026-05-19T00:29:12.573Z, votes: {'canUpvote': True}):
  You can generate another 100-200 test case using arcgen，then you will find some case not pass.

- **Chris Deotte** (2026-05-18T22:51:09.983Z, votes: {'canUpvote': True}):
  This generally means that your onnx file does not successfully solve the hidden test tasks. So your onnx hasn't found the true solution to solve the train data.

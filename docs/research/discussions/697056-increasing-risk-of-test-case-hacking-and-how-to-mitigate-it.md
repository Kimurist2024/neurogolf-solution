# Increasing Risk of Test Case Hacking and How to Mitigate It

- Topic ID: 697056
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/697056
- Author: keymoon (@keymoon)
- Posted: 2026-05-05T05:16:57.828167400Z
- Votes: 1
- Total messages: 1

## Body

With the new scoring scheme, I think the risk of test case proving / extracting has become higher. You can create a network to change score based on hidden case, and extract how the hidden test case looks like. This risk already existed before, but it will probably become much more viable now.

In particular, for tasks where the output is 1x1, it might be possible to get a very low score by hashing the input and using a lookup table.

Also, many tasks has the short solutions that work only for 99% of the time. If you want to reduce this kind of "99% solution", current "hidden test" approach is not enough. One possible approach would be to run a large number of extra ARC-GEN cases after the competition, maybe around 10,000 cases, and give 0 points to submissions that fail any of them.

To avoid concerns about the tests being adjusted after the competition, the organizers could generate these cases in advance, publish the hash of the ZIP file before the end of the competition, and then release the ZIP itself after the competition.

I think this competition is getting more and more fun these days, and this could help make it even better. Would you consider something like this?

## Comments (1)

- **Kameron Kilchrist** (2026-05-07T05:43:12.437Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  One issue is that at least some of the arc gen resulting inputs have ambiguous solutions, so a held out set may not be fair without additional filtering. At least with the current approach, it is possible to determine whether a particular solver clears the bar for the held out set. 
  
  I have at least one network that fails, e.g., 0.1% of arc gen and receives points via the scorer. That is an example of a problem with ambiguous solutions based on the published arc gen code.

# The Metric is Half the Battle

- Topic ID: 694914
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694914
- Author: jazivxt (@jazivxt)
- Posted: 2026-04-27T17:27:52.515811100Z
- Votes: 2
- Total messages: 5

## Body

Tasks like 110 and 191 have many ghost failures in the Metric that do not get reported as failures but yield a zero score.  Even the AI systems have a hard time getting creative to catch the nuance differences between a valid submission and an invalid one.  In these cases it mentions the densities of the networks, undeclared headers or possibly type declarations that may cause an issue but feel like whack a mole because one solution breaks another valid one.  Hope the Leader Board update and fix can solve this in the meantime, hoping for the best but always preparing for the anything.

![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWk4bWJ5czdjMXcyM20zeXUxanAwOHJjOWw4YTA5aHBjNXp2emExOCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/lRRjGTRlFwmQYFmmpU/giphy.gif)

## Comments (5)

- **hengck23** (2026-04-27T23:19:17.983Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Spend more time to handcraft template and let ai to complete. Or use hand craft solution for ai to learn

  - **jazivxt** (2026-04-28T00:22:32.703Z, votes: {'canUpvote': True}):
    Great advice, trying that right now but RTX 3070 is very limited for code LLMs and Free APIs only have so many tries before out of tokens when large contexts are shared, what would you suggest for a light solution @hengck23?

- **Michael D. Moffitt** (2026-04-28T22:38:29.277Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  The [newest metric](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230) is now live, and the rescoring of submissions is happening very soon.  We are hopeful this resolves any remaining ambiguities, and will be listening to the community for confirmation that all critical exploits have been resolved.
  
  Thanks again for the detailed feedback!!

- **Ali** (2026-04-27T21:42:57.567Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Tasks like 110 and 191 are hard to solve; my current score for both together is 10.89 (no tricks), will need optimization, but at least they are not zero.

  - **jazivxt** (2026-04-27T22:28:06.703Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Its impressive the community has solved all 400 in a matter of days, easy and hard alike. Its interesting the Kaggle community could solve almost any issue their competition sponsors trow at them.  This is testament to that, just the onnx speak on the forums alone is impressive not to mention what takeaways AI is having from the human interaction because of this competition.  Similar to code condensing any progress made that can make AI models smaller for faster inference and throughput is a win win for all.  If Kaggle partnered with OpenAI, Google or Claude to provide an API to the community they would be the beneficiaries of the Human to Machine rich interaction that this community brings. @asalhi great work look forward to seeing you in the top of the Leader Board!

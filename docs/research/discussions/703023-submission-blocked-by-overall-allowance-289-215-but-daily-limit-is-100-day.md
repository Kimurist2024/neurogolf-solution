# Submission blocked by "overall allowance (289/215)" but daily limit is 100/day

- Topic ID: 703023
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703023
- Author: ADARSH REDDY B (@adarshreddyb)
- Posted: 2026-05-27T22:18:04.731499700Z
- Votes: 18
- Total messages: 12

## Body

Hi all,
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10247677%2Ff06301350a74a1d31468603a46bfa967%2FScreenshot%202026-05-27%20150521.png?generation=1779920257552291&alt=media)
I'm running into a confusing submission error and wanted to flag it in case others are seeing the same thing.
When I try to submit, I get this message:
"Cannot submit
Your team has used its overall Submission allowance (289 of 215), please try again tomorrow UTC."

A few things don't add up:
1. The competition's stated daily submission limit is 100/day, not 215.
2. The "289 of 215" figure appears to be a lifetime/overall count, not a daily count — **289 is the total submissions I've made across the competition so far.**
3. The submission modal also shows "0 submissions remaining today. This resets in 2 hours" — which suggests the daily counter is what should be governing this, and it should reset shortly.

So the error message seems to be conflating an overall cap (215) with the daily limit (100), and the numbers shown (289 overall vs. 215 "allowance") don't match either documented limit. It's unclear whether:
- The daily counter is being displayed/enforced incorrectly, or
- This is a UI bug where the wrong limit message is being shown.

Has anyone else hit this? And could a moderator clarify. (Screenshot attached for reference)
Thanks!

## Comments (12)

- **Chris Deotte** (2026-05-28T11:20:44.257Z, votes: {'totalVotes': 12, 'canUpvote': True, 'totalUpvotes': 12}):
  @addisonhoward @inversion @sohiermse 
  
  Can you please fix submissions? It appears that the daily limit was accidentally changed to 5 a day instead of 100 a day. Now anyone with lots of submission cannot submit because their total exceeds `5 * days since beginning`.

  - **Adithya Giridharan** (2026-05-28T18:32:15.500Z, votes: {'canUpvote': True, 'totalUpvotes': 1}):
    is this solved now? been facing this myself too..

    - **Chris Deotte** (2026-05-28T19:27:49.157Z, votes: {'canUpvote': True}):
      It's solved now

- **Michael D. Moffitt** (2026-05-27T23:52:25.387Z, votes: {'totalVotes': 13, 'canUpvote': True, 'totalUpvotes': 13}):
  I’ll alert the Kaggle staff ASAP, as far as I know there were no planned changes to that limit.

- **Fritz Cremer** (2026-05-27T22:56:59.323Z, votes: {'totalVotes': 7, 'canUpvote': True, 'totalUpvotes': 7}):
  Hmm, I also have this issue and it seems to be due to a reset of the daily submission limit to 5. Kaggle enforces an overall submission limit of days_since_start * submission_limit to prevent teams that merge late from having had many more than 5 submissions per day on average. Since days_since_start = 43 and the default submission limit is 5, this matches perfectly: 43 * 5 = 215. Not sure why it was reduced but in any case, I hope if the host plans to actually plans to keep this limit, it will be handled properly with respect to the previous submissions made.

- **Ashley Oldacre** (2026-05-28T16:42:09.170Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
  Hello, we apologize for the disruption. The competition has been reset to 100 submissions a day. Thank you for bringing this to our attention.

- **Cona** (2026-05-28T03:28:22.090Z, votes: {'totalVotes': -1, 'canUpvote': True}):
  Seems to be a bug only affecting some people? I cannot submit any zips after 00:00 utc, while it tells me I have 0 remaining today. But there are also successful LB updates just in the last few minutes.

  - **hongan** (2026-05-28T03:33:48.300Z, votes: {'canUpvote': True}):
    it seems to only block those who have total submission above 215 times

    - **ADARSH REDDY B** (2026-05-28T03:46:49.573Z, votes: {'canUpvote': True}):
      Apparently now its 220😂😅

    - **Cona** (2026-05-28T04:37:59.267Z, votes: {'canUpvote': True}):
      True. The only thing I can do is to wait for Kaggle's debugging or for another 133 days🥹

- **hongan** (2026-05-27T22:40:39.457Z, votes: {'canUpvote': True}):
  i am also getting this from cli with only 3 submission today, when i tried to submit the fourth time: Submission not allowed: Your team has used its overall Submission allowance (521 of 215), please try again tomorrow UTC (84 minutes from now). @mmoffitt

- **(unknown)** (2026-05-28T00:01:08.003Z, votes: {}):
  (deleted)

# Can we have a thread to discuss about the patterns? 

- Topic ID: 695674
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695674
- Author: vishnuvardhan33 (@vishnuvardhan33)
- Posted: 2026-04-30T04:58:10.076270200Z
- Votes: 0
- Total messages: 4

## Body

I am stuck on 138, I need help. Maybe it's a good thing if we help each other out?![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17438398%2F65fcfad00b869b9a4399a32becb6a2eb%2FScreenshot%202026-04-30%20102743.png?generation=1777525082018223&alt=media)

## Comments (4)

- **hengck23** (2026-04-30T05:34:18.120Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  you go to arc-gen github website. they have the generator code.
  
  for this task138  
  - use corner detection the find the 4 corner to crop as output  
  - if the interior is gray pixel, then "snap" to gray edge. if it is green, "snap" to  green edge  
  - think of cumulative OR,  or cumulative ADD in snapping
  - maybe : reduce max (pixel * location)

  - **vishnuvardhan33** (2026-04-30T06:08:55.957Z, votes: {'canUpvote': True}):
    Thank you, I got it

- **jazivxt** (2026-04-30T05:08:33.097Z, votes: {'canUpvote': True}):
  When working with the LLM on an individual task it was helpful to provide the short Code Golf reference solver for the task which helps it understand the problem space better: 
  
  [138] 5daaa586.json
      detect_grid
      crop
      draw_line_from_point
      direction_guessing
  
  p=lambda g,k=35:-k*g or p([[g:=[e,g][l[0]==g>e<1<9>k]for e in l[::-1]]for*l,in zip(*g[0in g[0]:])],k-1)
  
  https://www.kaggle.com/code/jazivxt/system-control-pannel
  https://www.kaggle.com/code/jazivxt/oh-barnacles

  - **vishnuvardhan33** (2026-04-30T05:20:03.907Z, votes: {'canUpvote': True}):
    I just understood it. I'm so slow😅

# A cost effective way to use AI for this competition

- Topic ID: 694287
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694287
- Author: NNMax (@ashok205)
- Posted: 2026-04-24T09:10:27.256085200Z
- Votes: 12
- Total messages: 11

## Body

I'm seeing a lot of people use expensive AI subscriptions to squeeze out all the score they can get. Most seems to use codex and claude models predominantly which really helps for very fast iterations. As a student myself, I don't have any of these subscriptions except the student pack from github copilot which is so bad right now as they are imposing heavy rate limits as well. 

So here's an step by step almost no cost way that I use to solve the tasks.

## **Prerequisites** 
- Gemini (free), Deepseek (free), Qwen (free), Chatgpt (free), Github copilot Student pack (I dont think they are accepting the sign ups anymore).
- Agentic workflow with a free model that's good in coding ,example: openrouter free models, kilo code free models, opencode cli with free models, gemini cli. 
- If you can afford to run local model such as qwen 3.6 27B that's even better.

## **Workflow**
1. **Infer the rule by yourself** -> Visualise the task pairs and see the tasks for yourself. Almost all of the tasks are pretty much very easy for a human to crack.
2.  **Rule Format** -> Once you identify the rule write it down in a clear text file like .md or .txt with a one liner, constraints and rules. This should look like a leetcode question format where they give you the question and explanation, then rules and constraints.
3.  **Take three photos** -> One for training pairs, one for test pair, another for arc gen pair. Make sure these photos have different edge case pair for better decoding.
4. Paste the images, your inferred rule (as a plain prompt), taskxxx.json file to gemini and ask it to validate the rule by visually inspecting the image. After validation, ask it to give a structured one liner, constraints, rule with time/space complexity in mind.
5. You can now cross verify this using other free models like chatgpt, deepseek, qwen, etc as well.
6. Now in your agentic workflow, create a file named agent.md (anyname of your choice). This is the most important file, it should contain all details of the competition, evaluation criteria, exact version of the validation environment, banned ops, poisoned ops, important insights from discussions, etc. And importantly make sure to create a validate.py file to validate your models locally. 
7. Now attach the agent.md , rule.txt, taskxxx.json to the agentic ai model that you have access to. The reason why I'm suggesting agentic AI is that it has shell access to your project locally and can iterate on it's own.
8. Now for the prompting part. Write detailed instruction prompt that tells the agent what is the goal, what to do, how to do. Also impose some strict instructions. Web search is also a very important thing that you need to give your agent access to. As I'm using copilot, the only models that I can use reliably without rate limits are gpt 5.4 mini and claude haiku 4.5. These models are not great at finding rules or optimizing things but if you instruct them in a clear way, they can execute the flow very quickly.

## **Context window is important**
- Do not use unnecessary mcp tools. My only suggestions are a web search mcp like fetch/duckduckgo and desktop-commander (even this is optional)
-  Web chats always have lower context length. So create a new chat for every task. Same goes for agentic AI as well. New chat is always better than compacted conversation because of fresh context window.
 
I didn't include subagent usage as it will be sometimes useful and sometimes not. Really depends on the orchestrator agent.

For the optimizing part, there's two way for this. Either spend a lot of $$$ on frontier models and let them optimize your tasks or you can infer the onnx graph by yourself and compress it without breaking the logic. You only need to know how the ops work. Obviously this cost effective method consumes a lot of time but also lets you learn a lot about how to use multiple LLMs along with your own creativity to collectively solve a single task with much more control.

Would love to see how others are handling this as well, if anyone’s up for sharing.

## Comments (11)

- **yuanzhe zhou** (2026-05-13T03:40:57.180Z, votes: {'canUpvote': True}):
  I believe some participants have almost infinite access to gpt5.5/claude4.7+ and they do not care about the cost ... 👀

  - **Chan Kha Vu** (2026-05-13T03:58:58.807Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    On the contrary, GPT5.5 Claude4.7 are begging for access to THEM... the green company 😁

  - **NNMax** (2026-05-13T04:30:44.547Z, votes: {'canUpvote': True}):
    Yeah that's true, i saw someone who burnt 1B tokens in just a matter of days.

- **Gengsr** (2026-05-12T10:17:07.210Z, votes: {'canUpvote': True}):
  Hello, I'm a sophomore undergraduate student. I'd like to ask if you spent a lot of time training for this task? I completely handed it over to the AI and provided my own ideas for it to implement, but for a long time there were no results or very few tasks were solved. Is it because my prompt words were not good or do I need to switch to a different intelligent agent?

  - **Gengsr** (2026-05-12T10:18:47.043Z, votes: {'canUpvote': True}):
    I completely agree with you; that is why I use DeepSeek—it is very affordable for students.

    - **NNMax** (2026-05-12T15:56:50.470Z, votes: {'canUpvote': True}):
      Deepseek v4 is a very capable model. But you need to give it very detailed instructions. I found that it has a tendency to overthink a lot stuff which makes it sometimes unreliable when you want it to experiment with feedback based coding.

    - **Gengsr** (2026-05-12T16:03:44.853Z, votes: {'canUpvote': True}):
      Yes, may I ask how long you have been working on this to achieve such a high score? I would be very grateful.

    - **NNMax** (2026-05-12T16:47:44.447Z, votes: {'canUpvote': True}):
      I was working from the very start of the competition.

- **hwe owe** (2026-04-26T13:05:04.050Z, votes: {'canUpvote': True}):
  hi,can i team with you?

  - **NNMax** (2026-04-26T14:12:31.397Z, votes: {'canUpvote': True}):
    Sry, I already joined a team.

    - **hwe owe** (2026-04-27T09:08:49.730Z, votes: {'canUpvote': True}):
      ok,thanks

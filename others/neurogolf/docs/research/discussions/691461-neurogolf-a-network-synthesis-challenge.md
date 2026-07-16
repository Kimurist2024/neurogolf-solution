# NeuroGolf — A Network Synthesis Challenge

- Topic ID: 691461
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/691461
- Author: Michael D. Moffitt (@mmoffitt)
- Posted: 2026-04-14T20:45:21.664270400Z
- Votes: 15
- Total messages: 12
- Pinned: yes

## Body

Hello Kagglers,

We're thrilled to announce the launch of the **2026 NeuroGolf Championship**!

This three-month contest—featured in the [IJCAI-ECAI 2026](https://2026.ijcai.org/) Competition Track—adds a new spin to the classic challenge of code golf.  For those new to it, [code golf](https://en.wikipedia.org/wiki/Code_golf) is a programming competition where contestants solve a suite of tasks using the shortest possible source code (*see also: the championship Kaggle hosted [last year](https://www.kaggle.com/competitions/google-code-golf-2025)*).  In this new challenge, you'll instead design [neural networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) that exhibit the simplest possible structure, using any number of optimization techniques (e.g., [conventional ML training algorithms](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), [LLM-powered agentic coding](https://en.wikipedia.org/wiki/AI-assisted_software_development), [neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search), etc.) that you choose.

The tasks for this challenge are all drawn from the [Abstraction and Reasoning Corpus](https://arcprize.org/arc-agi) (ARC-AGI), a dataset first introduced by François Chollet in 2019 and later used in [many](https://www.kaggle.com/competitions/arc-prize-2024) [other](https://www.kaggle.com/competitions/arc-prize-2025) [competitions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2). Each puzzle requires a unique composition of transformations, so success will ultimately depend on your ability to create clever & efficient solutions.

To help you get started, we've created a [Kaggle notebook](https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship) where you can load individual ARC tasks and test out your networks.  Need help deciding which puzzles to tackle first? If so, we recommend starting with tasks [#6](https://arcprize.org/tasks/0520fde7), [#95](https://arcprize.org/tasks/4258a5f9), [#127](https://arcprize.org/tasks/54d9e175), [#261](https://arcprize.org/tasks/a79310a0), and [#331](https://arcprize.org/tasks/d364b489).  Finally, refer to the [main contest page](https://www.kaggle.com/competitions/neurogolf-2026) for instructions on how to zip up your networks and submit them for scoring.

Happy golfing! 🏌️

*[------- I'll update the space below with any relevant follow-ups -------]*

- **[April 21st]** [NeuroGolf Update for April 21st](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711)
- **[April 24th]** [NeuroGolf Update for April 24th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3448225)
- **[April 28th]** [NeuroGolf Update for April 28th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230) ... check out our [Metric Migration Manual](https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-metric-migration-manual)!
- **[April 29th]** [NeuroGolf Update for April 29th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230#3450385)
- **[April 30th]** [Metric updated](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230#3450385) to address two issues reported by @pavelsavchenkov and others.  Batch rescoring complete!
- **[May 1st]** We've set up `neurogolf.2026@gmail.com` so that teams can reach out to the organizers privately with any sensitive questions.
- **[May 2nd]** [NeuroGolf Update for May 2nd](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696377#3452306)
- **[May 4th]** [NeuroGolf Update for May 4th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953)
- **[May 5th]** [NeuroGolf Update for May 5th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953#3453590)
- **[May 14th]** [NeuroGolf Update for May 14th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/699562#3457881)

## Comments (12)

- **Hsyn Kskn** (2026-05-04T12:21:50.053Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Dear Organizers,
  
  I am writing to bring to your immediate attention a set of severe vulnerabilities in the official evaluation pipeline of the 2026 NeuroGolf Championship. The current scoring mechanism, which relies on the onnx-tool library (v1.0.1) to compute network cost (parameters + memory + MACs), contains multiple bugs that allow functionally correct networks to achieve artificially near-zero costs. This completely undermines the competition’s goal of “designing the smallest possible neural networks” and rewards exploit engineering over genuine efficiency.
  
  Below I summarize the discovered exploits with concrete evidence from the attached notebook (onnx-tool-exploit-report-apr-25-v2.ipynb). All tests were run on task104 (public ARC-AGI task) using the official environment.
  
  Summary of Exploits & Impact
  Exploit	Affected Operator(s)	Incorrect Behavior	Achieved Cost (task104)	Correct Cost (approx.)
  Compress + Resize	Compress, Resize (nearest)	Compress output shape becomes [0,10,30,30] due to fake zero tensors; Resize MACs = 0.	72	> 850,000
  Compress + Gather	Compress, Gather	Same Compress bug; Gather MACs = 0.	96	> 850,000
  Compress + GridSample	Compress, GridSample	Same Compress bug; GridSample MACs = 0.	112	> 850,000
  TopK	TopK	No MACs counted for ranking operation.	9,104	~ same as Gather
  Where (broadcast)	Where	Shape inference fails to broadcast → output shape [1,10,1,30] instead of [1,10,30,30].	3,600	> 36,000
  ConstantOfShape	ConstantOfShape	Data‑dependent shape evaluated on fake zeros → output shape [0,0,0,0] → zero memory.	1,286	> 12,000
  Most striking: A correct network for task104 can be built with Resize + Compress that the evaluator scores as cost = 72 (score ≈ 20.72). In contrast, the example network in the competition overview (single 3×3 conv) costs 850,500 (score ≈ 11.39). This is a ~11,800x reduction achieved by exploiting bugs, not by efficient design.
  
  Technical Evidence (from the notebook)
  1. Compress bug – output shape vanishes
  The Compress operator selects elements based on a condition. During shape_infer(None), onnx-tool replaces unknown tensors with all‑zero synthetic tensors. The condition then becomes all false, so the node is profiled with out=[0, 10, 30, 30] and memory = 0:
  
  python
  # From the notebook – Compress output shape
  # Expected: [1, 10, 30, 30] (one full board)
  # Actual from onnx-tool:
  'select_bank': {'out': [0, 10, 30, 30], 'memory': 0}
  Relevant source code locations (annotated in notebook):
  
  graph.py L1023‑L1035 (fake zero injection)
  
  tensor.py L433‑L438
  
  CompressNode (node.py L1908‑L1915)
  
  2. Where broadcast shape failure
  ONNX Where uses broadcasting. Input shapes [1,10,1,30] and [1,10,30,1] should produce [1,10,30,30]. However, onnx-tool calls _max_shape() which cannot generate a broadcasted shape larger than any input. Result:
  
  python
  # Where node profiled as:
  'out': [1, 10, 1, 30]   # only a thin slice, memory 1200 instead of 12000
  Code: node.py L80‑L91 (_max_shape) and L779‑L785 (WhereNode.shape_infer).
  
  3. TopK – zero MACs for sorting
  TopKNode has no custom profile() method, so it falls back to the base implementation that reports macs = 0. The ranking of 4 values (which should cost at least a few operations) is ignored entirely.
  
  4. ConstantOfShape – dynamic shape evaluated on fake zeros
  The node’s output shape depends on an input tensor (runtime_shape). During shape inference, onnx-tool computes runtime_shape using fake zeros, leading to [0,0,0,0] and zero memory cost.
  
  Why This Breaks the Competition
  Fairness – Participants who discover these bugs can achieve orders‑of‑magnitude lower costs than those who build genuinely small networks.
  
  Goal misalignment – The scoring no longer measures “how much computation these tasks actually require”. Instead, it measures “how well can you trick the profiler”.
  
  Reproducibility – The official starter notebook (the-2026-neurogolf-championship.ipynb) uses a different profiling approach (via model_profiler.py) and shows honest costs. This creates a false sense of the scoring mechanism.
  
  Irrelevance of constraints – The 1.44MB ONNX file limit and the banned operator list (Loop, Scan, etc.) do not prevent any of the shown exploits.
  
  Recommended Actions
  To preserve the integrity of the competition, I strongly suggest one or more of the following:
  
  Immediately patch onnx-tool – Fix shape inference to handle data‑dependent shapes correctly (do not use fake zeros for shape‑determining inputs). Add missing MAC counts for operators like TopK, Gather, Resize (nearest), GridSample.
  
  Replace the profiler – Use a more robust tool (e.g., onnxruntime + custom flop counter, or keras-flops, ptflops-like tools for ONNX) that does not rely on fragile static analysis.
  
  Disable offending operators – Temporarily ban Compress, ConstantOfShape, Where, TopK, Gather, Resize, GridSample until a fix is released. (Though this would limit legitimate solutions.)
  
  Re‑evaluate all submissions – If a patch is deployed after the competition starts, re‑run the scoring on all submitted models to ensure fairness.
  
  I am happy to provide the full notebook, minimal working examples of each exploit, and assist in testing any patches.
  
  Thank you for your attention to this critical matter. The NeuroGolf Championship is an exciting and important contest; fixing these issues will ensure it remains a true benchmark for efficient neural computation.
  
  Sincerely,
  [Your Name / Team Name]
  [Contact Info – Kaggle username, email]
  
  Attachments (provided):
  
  onnx-tool-exploit-report-apr-25-v2.ipynb (full reproduction of all exploits)
  
  (Optional) Minimal ONNX files for each exploit

  - **Michael D. Moffitt** (2026-05-04T21:29:57.200Z, votes: {'canUpvote': True}):
    Thank you, Hsyn -- this is an invaluable notebook, and it will go a long way toward identifying areas of improvement in the third-party profiler.
    
    Because these issues were so pervasive and painful for teams, we've decided to (a) drop MACs from the objective completely, and (b) rely exclusively on the official ONNX Runtime for precise shape analysis (see [today's announcement](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953)).  Our hope is that this will dramatically reduce the need for teams to rely on bugs & exploits to juice their network scores.

- **Tsitsino Dzotsenidze** (2026-04-19T07:59:49.477Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Hello, thank you for organizing this competition.
  
  I have one eligibility question. I am currently working on ARC Prize 2026 and wanted to ask whether it would be permitted to also participate in The 2026 NeuroGolf Championship at the same time, or whether that would conflict with the rules for ARC Prize 2026 or related competitions.
  
  I would appreciate any clarification before joining or submitting.
  
  Best regards,
  Tsitsino Dzotsenidze

  - **Ashley Oldacre** (2026-04-20T21:59:29.817Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Hello Tsitsino Dzotsenidze, there are no problems with competing in both the The 2026 NeuroGolf Championship and the ARC Prize 2026 competition. Thank you for the question and good luck!

- **Warut t** (2026-04-16T09:03:46.690Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Hmm Maximum 100 mulligans a day

- **yuanzhe zhou** (2026-04-20T04:06:34.480Z, votes: {'canUpvote': True}):
  All the test dataset is provided in the data section?  How do we evaluate the final results?

  - **yuanzhe zhou** (2026-04-30T22:53:40.780Z, votes: {'canUpvote': True}):
    @mmoffitt Do we have a private LB here?

    - **Michael D. Moffitt** (2026-04-30T23:02:06.313Z, votes: {'canUpvote': True}):
      No private LB, just the public one (so, no risk of a post-contest shakeup).
      
      Note that we've had a couple metric updates so far, each one accompanied by a LB refresh.  Teams have proven to be quite adept at creating networks with unanticipated properties!

    - **Suman Sharma** (2026-04-30T23:15:35.577Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      What is the newest update could u attach a new thread to have all the bugs and updates in one place it would be really helpful

    - **Michael D. Moffitt** (2026-04-30T23:36:26.830Z, votes: {'canUpvote': True}):
      @suman2208 Great suggestion -- I've edited the original post up above to include a link to all updates, and will continue to do so for the rest of the contest.

    - **Suman Sharma** (2026-04-30T23:43:19.300Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Thanks a lot for the prompt response really appreciate the effort

- **vineet kumar** (2026-04-19T14:05:55.927Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  thank you for this....

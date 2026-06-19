# The highest single task score is 25 (fixed)

- Topic ID: 693247
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247
- Author: leo (@calibrator)
- Posted: 2026-04-20T11:29:26.439252500Z
- Votes: 30
- Total messages: 10

## Body

## Update: The exploits have been fixed on 21st April.
---

A combination of several bugs in `onnx_tool` can make every task reach 25. 

To achieve 25, we need cost = MACs + memory + num params = 1 exactly. @kq5yyy and @shinh0 already reported several bugs/exploits in https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827 that can reduce some of them, but still impossible to reduce the total cost to 1. What if we can make one of them to be negative? 

By chaining the issues below, memory can be turned into negative:

- [`PadNode.shape_infer`](https://github.com/ThanatosShinji/onnx-tool/blob/43b5c5e4cdff13038cc23e3655f3a2e6b821b75d/onnx_tool/node.py#L1408-L1445) allows to output negative shapes. 
- [`get_memsize`](https://github.com/ThanatosShinji/onnx-tool/blob/43b5c5e4cdff13038cc23e3655f3a2e6b821b75d/onnx_tool/tensor.py#L455-L456) calculates the mem size of a node by simply taking product of the shape from each dimension. 
- The total mem size of a network is the sum of mem size of all nodes.

As reported by @kq5yyy , there is a divergence between runtime and static analysis, i.e. shape_infer evaluates value-dependent shapes with zero-filled dummy inputs. Taking advantage of this, we can make Pad's pads tensor be runtime valid (all zeros) but static negative for analysis by onnx_tool. Wiring the Pad node as an orphan (its output isn't consumed by anything else in the graph) keeps the model's real output intact while still letting onnx_tool profile the node and absorb its negative contribution. Then all we need to do is to adjust this negative constant to make the cost = 1. 

I publish an example to prove the theory: https://www.kaggle.com/code/calibrator/neurogolf-bug-demo-notebook?scriptVersionId=313065924

I'm not using this in my submissions. Escalating so the host and the onnx_tool maintainer can address this issue soon.

@mmoffitt @kevinyuluo

## Comments (10)

- **Michael D. Moffitt** (2026-04-21T23:06:47.273Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  We've fixed this by imposing the following check (will work with any version of `onnx-tool`):
  
  ```python
    for key in g.nodemap.keys():
      if g.nodemap[key].memory < 0:
        print(f"Error: Negative memory value detected.")
        return None, None, None
  ```
  
  Please let us know if any other issues crop up.  Thank you!

  - **leo** (2026-04-22T00:06:26.797Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    Cheers for the fix! I'll see if we can flip MACs or Params to negative. 🙃

    - **Michael D. Moffitt** (2026-04-28T21:59:36.193Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      > I'll see if we can flip MACs or Params to negative.
      
      It was indeed possible 😉 but we've fixed that too in our [latest update](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230).

    - **leo** (2026-04-28T22:21:00.350Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Yes, negative MACs is the new trick I found. I believe num of params is safe, but i can’t be certain. Glad to hear this has been addressed!

    - **Michael D. Moffitt** (2026-04-28T22:24:19.480Z, votes: {'canUpvote': True}):
      We nearly missed it.  Thank you for thinking through these possibilities!

- **Michael D. Moffitt** (2026-04-20T18:27:50.800Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  Great find!!  We'll have an update soon, stay tuned.

  - **Rafaël Labbé** (2026-04-20T21:25:32.850Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    What is the policy if someone saves a bug like that until the very ending and then makes a last-day submission with a weird bug that gives them a massive score advantage? 
    
    I (and I imagine more people with me) are currently hesitant on spending effort in this competition due to the risk of losing to something unpredictable like that.

- **Navneet** (2026-04-21T06:26:18.397Z, votes: {'totalVotes': -2, 'canUpvote': True}):
  Cool high single task score @calibrator

- **Yu Luo** (2026-04-21T12:26:25.810Z, votes: {'canUpvote': True}):
  I don't know the details of score. But the profiled result of your model looks good to me: 
  ```python
  Name           Type       Forward_MACs    FPercent    Memory    MPercent      Params  PPercent    InShape     OutShape
  -------------  ---------  --------------  ----------  --------  ----------  --------  ----------  ----------  ----------
  Greater_143    Greater    0               0.00%       0         0.00%              0  0.00%       0           0
  MaxPool_36     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_101    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_121    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_106        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_93     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_56     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_38         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_50         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_23         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_129    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_117    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_76     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_32     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_29     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_115        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_132    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_99         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_51         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_98         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_61     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_82         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_27         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_33     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_111        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_128    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_15         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Sub_11         Sub        900             0.37%       3,604     0.71%              1  0.11%       1           1x1x30x30
  Max_70         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_109    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_83         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_123        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_92     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_49     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_81     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_41     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_85     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Sub_12         Sub        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_52     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_133    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_73     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_120    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_108    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_75         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_60     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_125    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_107        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_35         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_110        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Add_140        Add        9,000           3.75%       36,000    7.14%              0  0.00%       1x10x30x30  1x10x30x30
  MaxPool_100    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_86         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_94         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_20     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_87         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  ReduceMax_142  ReduceMax  10              0.00%       0         0.00%              0  0.00%       1x10x1x1    0
  Mul_71         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_34         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_40     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_57     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_24     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_31         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_88     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_69     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_26         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_127        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_130        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_89     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_131        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  ReduceSum_10   ReduceSum  9,000           3.75%       3,600     0.71%              0  0.00%       1x10x30x30  1x1x30x30
  Max_118        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_114        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_43         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_124    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_67         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_79         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Where_139      Where      0               0.00%       3,680     0.73%             20  2.17%       1x1x30x30   1x1x30x30
  Max_78         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_119        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_63         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_22         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_16     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_59         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_47         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_64     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_53     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_91         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_95         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Sub_13         Sub        900             0.37%       3,600     0.71%              0  0.00%       1           1x1x30x30
  MaxPool_21     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_19         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_25     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_66         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_74         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_58         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_18         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_42         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_113    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_122        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_62         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_28     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_44     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_104    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_14         Max        900             0.37%       7,200     1.43%            900  97.72%      1x1x30x30   1x1x30x30
  Max_126        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_103        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_54         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_39         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_45     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_116    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_102        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_17     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_37     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Where_145      Where      0               0.00%       4         0.00%              0  0.00%       0           1
  Mul_55         Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_134        Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_46         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_105    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_137        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_48     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_80     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_96     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_84     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Sub_136        Sub        900             0.37%       3,600     0.71%              0  0.00%       1           1x1x30x30
  MaxPool_97     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_68     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_77     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_72     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Mul_135        Mul        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_30         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_112    MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Max_90         Max        900             0.37%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  MaxPool_65     MaxPool    2,700           1.12%       3,600     0.71%              0  0.00%       1x1x30x30   1x1x30x30
  Total          _          240,310         100%        504,088   100%             921  100%        _           _
  ```

  - **leo** (2026-04-21T23:52:32.357Z, votes: {'canUpvote': True}):
    Thanks for looking into this issue. 
    
    I got the same result with `onnx_tool.model_profile`. The scoring function of this competition uses the following code to calculate memory footprint.
    ```python
    def score_network(m):
      model = onnx_tool.loadmodel(m, {'verbose': False})
      g = model.graph
      g.graph_reorder_nodes()
      g.shape_infer(None)
      g.profile()
      # skipped some validation logics here
      return int(sum(g.macs)), int(g.memory), int(g.params)
    ```
    The output of this function gives me: MACs: 240310, Memory: -241233, Params: 924, which is different to the output of `model_profile`. However, the results can match if we set `hidden_ops=None` when calling `model_profile`.

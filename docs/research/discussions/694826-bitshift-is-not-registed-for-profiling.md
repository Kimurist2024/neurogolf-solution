# BitShift is not registed for profiling

- Topic ID: 694826
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694826
- Author: hengck23 (@hengck23)
- Posted: 2026-04-27T10:58:57.255253600Z
- Votes: 6
- Total messages: 6

## Body

BitShift are clever hacks and would be good if they can be used

/home/hp/app/anaconda3.11-cv/lib/python3.11/site-packages/onnx_tool/node.py:2829: UserWarning: node BitShift is not registed for profiling, return 0 Macs and 0 params as default. Use NODEPROFILER_REGISTRY to register your profiler for this node.
  warnings.warn(f'node {n.op_type} is not registed for profiling, return 0 Macs and 0 params as default. '
 
  File "/home/hp/app/anaconda3.11-cv/lib/python3.11/site-packages/onnx_tool/node.py", line 210, in value_infer
    raise NotImplementedError(f'this Node {self.op_type}-{self.name} has no value_infer')
NotImplementedError: this Node BitShift-BitShift_0 has no value_infer

## Comments (6)

- **Michael D. Moffitt** (2026-05-07T00:23:27.590Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Now that we've [eliminated the dependency on onnx-tool](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953#3454132), BitShift should be back on the menu!

  - **hengck23** (2026-05-07T03:54:41.873Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    That is great! Bitwise network open out a new class of solution!

- **MassimilianoGhiotto** (2026-04-28T06:17:31.047Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  I suggest reading this: https://www.kaggle.com/code/emanuellcs/neurogolf-2026-zero-mac-heuristics-dc-pp

  - **hengck23** (2026-04-28T09:52:04.587Z, votes: {'canUpvote': True}):
    These are actually not optimised. It didn’t exploit data structures. Most of them are translation from generator solution to onnx ops. Their scores are in 12 to 13 range. Hand optimisation scale are in 14 to 15 range and the worst i get is 13 range

    - **hengck23** (2026-04-28T09:57:26.593Z, votes: {'canUpvote': True}):
      as as example, 14.09 bitwise solution for task009. open public solution are about 11-12
      
      ```
      def make_task009():
          from spox.opset.ai.onnx import v18 as op
      
          SHAPE = (1, 10, 30, 30)
          input = argument(Tensor(np.float32, SHAPE))
      
          def op_slice(x, starts, ends, axes, steps):
              return op.slice(
                  x,
                  starts = op.constant(value=np.asarray(starts, dtype=np.int64)),
                  ends   = op.constant(value=np.asarray(  ends, dtype=np.int64)),
                  axes   = op.constant(value=np.asarray(  axes, dtype=np.int64)),
                  steps  = op.constant(value=np.asarray( steps, dtype=np.int64)),
              )
          def op_constant(value, dtype=np.int64):
              return op.constant(value=np.asarray(value, dtype=dtype))
      
          #todo: collect all common axes into one const parameter
      
          #---------------
          x = op.cast(input, to=np.uint8)
      
          occupancy = op_slice(
              x,
              starts=[ 1, 0, 2],
              ends  =[10,30, 3],
              axes  =[ 1, 2, 3],
              steps =[ 1, 1, 1],
          )
          occupancy = op.reduce_max(occupancy, axes=op_constant([1]))
          #.e.g 11111111110000 (1:occupied, 0: padding)
      
          gridcolor = op_slice(
              x,
              starts =[ 0,  0,  0],
              ends   =[10, 30, 30],
              axes   =[ 1,  2,  3],
              steps  =[ 1,  3,  3],
          )
          #linecolor = input[0,:,0,2]
      
          linecolor = op_slice(
              x,
              starts=[ 1, 0, 2],
              ends  =[10, 1, 3],
              axes  =[ 1, 2, 3],
              steps =[ 1, 1, 1],
          )
          linecolor = op.arg_max(linecolor, axis=1)
          linecolor = op.reshape(linecolor, shape=op_constant([1]))
          linecolor = op.add(linecolor, op_constant([1]))
      
      
          #----
          # color encode table
          color_table = [
              [0, 1, 2, 4, 8, 16, 32, 64, 128, 255] #for debug
          ]
          for lc in range(1,10):
              vals = []
              bit = 1
              for c in range(10):
                  if c == 0 :
                      vals.append(0)   #bg marker
                  elif c == lc:
                      vals.append(255) #linecolor marker
                  else:
                      vals.append(bit)
                      bit *= 2
              color_table.append(vals)
      
          color_table = op.constant(
              value=np.array(color_table, dtype=np.uint8).reshape(10, 10, 1, 1)
          )
          depth = op.gather(color_table, linecolor, axis=0)
      
          # tempopray depth for debug
          # depth0 = op.constant(value=np.array(
          #    [0, 1, 2, 4, 8, 16, 32, 64, 128, 0], dtype=np.uint8).reshape(1, 10, 1, 1))
      
          fg = op.reduce_max(op.mul(gridcolor, depth), axes=op_constant([1]), keepdims=True)
      
          # ---- connect point by propagation ----
      
          def shift_w(x, k):
              z = op.constant(value=np.asarray(0, dtype=np.uint8))
              if k>0:
                  y = op_slice(x, [0], [10-k], [3], [1])
                  pads = op_constant([0,0,0,k,0,0,0,0])
              else:
                  y = op_slice(x, [-k], [10], [3], [1])
                  pads = op_constant([0,0,0,0,0,0,0,-k])
              return op.pad(y, pads, constant_value=z, mode="constant")
      
          def shift_h(x, k):
              z = op.constant(value=np.asarray(0, dtype=np.uint8))
              if k>0:
                  y = op_slice(x, [0], [10-k], [2], [1])
                  pads = op_constant([0,0,k,0,0,0,0,0])
              else:
                  y = op_slice(x, [-k], [10], [2], [1])
                  pads = op_constant([0,0,0,0,0,0,-k,0])
              return op.pad(y, pads, constant_value=z, mode="constant")
      
          def propagate_h(x, d):
              y = x
              y = op.bitwise_or(y, shift_h(y, d * 1))
              y = op.bitwise_or(y, shift_h(y, d * 2))
              y = op.bitwise_or(y, shift_h(y, d * 4))
              y = op.bitwise_or(y, shift_h(y, d * 8))
              return y
      
          def propagate_w(x, d):
              y = x
              y = op.bitwise_or(y, shift_w(y, d * 1))
              y = op.bitwise_or(y, shift_w(y, d * 2))
              y = op.bitwise_or(y, shift_w(y, d * 4))
              y = op.bitwise_or(y, shift_w(y, d * 8))
              return y
      
      
          left = propagate_w(fg, 1)
          right = propagate_w(fg, -1)
          hfill = op.bitwise_and(left, right)
      
          bottom = propagate_h(fg, 1)
          top = propagate_h(fg, -1)
          vfill = op.bitwise_and(top, bottom)
      
          filled = op.bitwise_or(vfill,hfill) #1x1x10x10
          filled = op.reshape  (filled, op_constant([1, 1, 10, 1, 10, 1]))
          filled = op.expand   (filled, op_constant([1, 1, 10, 3, 10, 3]))
          expanded = op.reshape(filled, op_constant([1, 1, 30, 30]))
      
          #-- handle line ---
          line = op.constant(value=np.asarray([0,0,255, 0,0,255, 255,255,255], dtype=np.uint8).reshape(1,1,1,3,1,3))
          line = op.expand(  line, op_constant([1, 1, 10, 3, 10, 3]))
          line = op.reshape( line, op_constant([1, 1, 30, 30]))
          expanded = op.bitwise_or(expanded, line)
          #expanded = op.max([expanded, line])
      
          #handle occpancy ---
          occ_row = occupancy  # [1,1,30,1]
          occ_col = op.transpose(occupancy, perm=[0, 1, 3, 2])  # [1,1,1,30]
          inside = op.and_(op.cast(occ_row, to=bool), op.cast(occ_col, to=bool)) #1,1,30,30
      
          expanded = op.where(
              inside,
              expanded,
              op_constant([254], np.uint8),#invalid
          )
          expanded_bg = op.equal(expanded, op_constant([0], np.uint8))
      
      
          #final expansion to 30x30 -----
          expanded_colors = op.equal(
              expanded,
              op_slice(depth,[1],[10],[1],[1])
          )
          output = op.concat([
              op.cast(expanded_bg, to=np.bool),
              expanded_colors
          ], axis=1)
      
          model = spox.build(
              inputs={"input": input},
              outputs={
                  "output": output,
              },
          )
          onnx.checker.check_model(model)
          return model
      
      
      ```

- **hengck23** (2026-04-28T00:42:48.443Z, votes: {'canUpvote': True}):
  as an example, consider task009. lt has multiple varying colors, best handled as onehot data. but instead of 1/0 per byte channel, i want to use 1/0 per bit. i can cut the memory to 1/8 times. the opsetset supports it, but the profiler does not. I wonder can the host support this?
  
  i can still use arithmetic, but using bitwise op is more natural

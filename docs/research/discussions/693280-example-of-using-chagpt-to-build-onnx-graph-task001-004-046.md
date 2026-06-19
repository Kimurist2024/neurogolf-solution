# example of using chagpt to build onnx graph: task001/004/046

- Topic ID: 693280
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693280
- Author: hengck23 (@hengck23)
- Posted: 2026-04-20T13:50:22.923776900Z
- Votes: 13
- Total messages: 26

## Body

Let's consider task001. Baseline results is (mlp?) 13.6pts from public notebook :
https://www.kaggle.com/code/manish756/handcrafted-baseline  

##[1] Input the below as prompt. I ask him to write a proper specification.md so that i can try other coder like Claude or Gemini, etc...  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F516be0430c584209c1278360f184d080%2FSelection_2903.png?generation=1776692680992054&alt=media)

##[2] I ask him to code make\_onnx() function. Then I use kaggle verify\_network() to make score printout which i feedback to chatgpt. I ask for:
- identify and explain bottleneck
- strategy for improvement.  
- what is the theoretical limit  

##[3] I choose the best strategy and repeat.


i write no onnx code at all (except those to verify and draw results). Below is what chatgpt has coded:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F572c667f72ac36acbf17d88d3279214c%2FSelection_2922.png?generation=1776693020246414&alt=media)

## Comments (26)

- **hengck23** (2026-04-20T14:05:18.147Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  **Background:**
  
  - As a CV engineer, creating ONNX models was part of my job in the past. However, my main focus was on algorithm design and model training, so my ONNX knowledge was limited and mostly handled by other implementation engineers. Also, that was 8 years ago, and I haven’t worked with ONNX since.
  
  - I’m quite surprised by the code written by ChatGPT today—its strategy and level of understanding are impressive.
  
  - However, it still runs into problems when debugging. For example, it made a mistake in indexing for decomposition and seemed confused. I suggested that the easiest way to debug indexing is to use an `arange` array as input and build a specialised ONNX graph to catch the suspected bug. It seemed to understand that approach.

- **hengck23** (2026-04-20T15:16:43.133Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  let's try difficult task004. this prompt is good
  
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F99910d1f4aa1718e58fb28a6a8c81c29%2FSelection_2928.png?generation=1776698186494203&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F4ecc71b07270b2dc85d8d0e521ec9687%2FSelection_2929.png?generation=1776698201519079&alt=media)
  
  "to gpt: i think we implement in numpy array first to check logic. then we can implement onnx graph and optimsed later"

  - **hengck23** (2026-04-20T19:49:16.127Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    there is a shortcut after interacting with chatgpt. we just need to detect x,y or corner and freeze the whole row and col, while moving the rest.
    
    ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F67379010ebe26e03adf6c50d99055260%2FSelection_2935.png?generation=1776714322380726&alt=media)
    
    i also worked with FPGA inference engineer to design algorithm before. We had meeting and i would ask him  the optimium/preferred operations, mem bandwidth, etc ... Talking to chatgpt is like that. To release the full potential of AI coding, you need to ask the right magic question or give him the right magic instruction.

    - **hengck23** (2026-04-20T19:54:31.473Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3c1805788eea3d3d5b8dc69282fc7431%2FSelection_2937.png?generation=1776714870071201&alt=media)

    - **hengck23** (2026-04-20T20:10:31.233Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      np implementation. ask your chatbot to comment (on suitability to implement for onnx graph), then ask him to make onnx graph and optimised
      
      ```
      import numpy as np
      
      def shift_right_keep_size(arr: np.ndarray, fill_value=0):
          out = np.full_like(arr, fill_value)
          out[:, 1:] = arr[:, :-1]
          return out
      
      
      def find_bottom_right_corner_of_color(grid: np.ndarray, color: int):
          H, W = grid.shape
          for r in range(1, H):
              for c in range(1, W):
                  if (
                      grid[r, c] == color and
                      grid[r, c - 1] == color and
                      grid[r - 1, c] == color
                  ):
                      return r, c
          return None
      
      
      def build_keep_mask_shortcut_arith(grid: np.ndarray, color: int):
          """
          keep = row_r OR col_c
          Assumes overlap never happens.
          """
          H, W = grid.shape
          color_mask = (grid == color).astype(np.float32)
      
          keep = np.zeros((H, W), dtype=np.float32)
      
          corner = find_bottom_right_corner_of_color(grid, color)
          if corner is None:
              return keep
      
          r, c = corner
      
          row_mask = np.zeros((H, W), dtype=np.float32)
          row_mask[r, :] = color_mask[r, :]
      
          col_mask = np.zeros((H, W), dtype=np.float32)
          col_mask[:, c] = color_mask[:, c]
      
      
          keep = row_mask + col_mask
          keep[r,c]=1
          return keep
      
      
      def solve_task004_numpy_shortcut_arith(grid: np.ndarray):
          out = np.zeros_like(grid)
      
          colors = [v for v in np.unique(grid) if v != 0]
      
          for color in colors:
              color_mask = (grid == color).astype(np.float32)
              keep = build_keep_mask_shortcut_arith(grid, color)
      
              move = color_mask * (1.0 - keep)
              moved = shift_right_keep_size(move, fill_value=0.0)
      
              out += (keep * color + moved * color).astype(grid.dtype)
      
          return out
      
      
      
      
      
      ####################################################################
      if __name__ == "__main__":
          from my_utils import *
          from neurogolf_utils import *
      
          DATA_DIR = '/media/hp/8TB-HDD/work/2026/kaggle/neurogolf/data/neurogolf-2026'
          task_file = f'{DATA_DIR}/task004.json'
          with open(task_file, 'r') as f:
              task = json.load(f)
      
          # show_legend()
          # show_examples(task["train"])
          # show_examples(task["test"])
          # show_examples(task["arc-gen"])
          # plt.show()
      
          for i in range(10):
              print(i, '---------------------------')
              input = np.array(task["arc-gen"][i]["input"])
              truth = np.array(task["arc-gen"][i]["output"])
              predict = solve_task004_numpy_shortcut_arith(input)
              print(input.shape, truth.shape, predict.shape)
              correct = np.array_equal(predict, truth)
              print('correct', correct)
      
              if not correct:
                  print('input')
                  print_grid_color(task["arc-gen"][i]["input"])
                  print('truth')
                  print_grid_color(task["arc-gen"][i]["output"])
                  print('model output')
                  print_grid_color(predict.tolist())
      
      
      ```

    - **hengck23** (2026-04-20T20:34:52.543Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      prompt " does onnx have magic opset at high level, e.g. i know the magic shift_add fusion for ARM. what onnx op should be encouraged to reduce macs or mem"
      
      1. Gemm — the classic “magic op”
      2. Conv — more than convolution
      It can include:
      - bias addition
      - padding
      - stride
      - dilation
      4. Where — branch without branching
      5. Gather — memory-level trick
      6. Einsum — ultimate abstraction
      
      "how about memory planning tricks?"
      
      "any repo that i can find,  example of highly optimised/fused onnx graphs?"

- **hengck23** (2026-04-21T02:29:32.310Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  task004 public reference:
  https://www.kaggle.com/datasets/daphne4sg/claude-golf   
  It appears to require 3943800 MACs + 553592 bytes + 4298 params, yielding 9.680 points.

  - **Russell Kirk** (2026-04-21T02:36:00.713Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    I got 13.7 for task 4, if you want that as a data point.

    - **hengck23** (2026-04-21T03:36:12.460Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      i ask my chatpt to try harder first: "hi chatgpt, you have lost! human kaggler has  13.7 for task004. please re-evaluate the theoretical limit and send me report. do not leave the office today until work is done! 😡"

    - **hengck23** (2026-04-21T04:08:46.317Z, votes: {'canUpvote': True}):
      he said he has a solution after i give him some hint, let's wait ....
      
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3db55128915d9ecf7ef59c9ecfab218c%2FSelection_2961.png?generation=1776744386296410&alt=media)
      
      chatgpt is good at automatic micro improvement.  But for high jump, the solution needs to be reset and i think he needs help from human! 
      
      ---
      
      there is a limit we can get if we use certain operations like conv, etc. there is also a limit for processing in 2d or 1d etc.. once we know the score (achieved) by others, we/he can "roughly guess what operation etc should/could be used "

    - **Russell Kirk** (2026-04-21T04:11:40.357Z, votes: {'canUpvote': True}):
      I wrote my solution in Racket then I translated them into python.  I think Racket, in general, is much easier to think in.

    - **hengck23** (2026-04-24T23:11:46.337Z, votes: {'canUpvote': True}):
      @russcore 
      
      Instead of shifting the pixel, I can treat the slanted rect as pixel locations! Below is verified code
      
      ```
      def shear_slant_rect_from_bbox(mask):
          H, W = mask.shape
          ys, xs = np.where(mask)
          if len(ys) == 0:
              return np.zeros_like(mask)#mask.astype(np.int32)
      
          y1, y2 = ys.min(), ys.max() #if flatten, these are simply first and last pixel
          x1, x2 = xs.min(), xs.max()
      
          bh = y2 - y1 + 1 #bh>=3 from data
          bw = x2 - x1 + 1
      
      
          '''
          verfiied from experiment 
          print(y1,x1,y2,x2)
          print(bh,bw)
      
          7,9,4 # rw = bw-5  #5=bh-2
          4 6,4 # rw = bw-2  #2=bh-2
          4 7,5 # rw = bw-2
          
          3 5, 4 # rw = bw-1 #1=bh-2
          3 4, 3 # rw = bw-1
          '''
      
          # rectangle horizontal width after removing diagonal drift
          rw = bw - (bh - 2)
          if rw <= 0: #never happens!
              raise ValueError(f"bad geometry: bw={bw}, bh={bh}, rw={rw}")
      
          out = np.zeros_like(mask)
      
          # top L shifted right by 1
          out[y1,     x1 + 1 : x1 + rw + 1] = 1
          out[y1 + 1, x1 + 1] = 1
      
          # bottom L fixed
          out[y2,     x2 - rw + 1 : x2 + 1] = 1
          out[y2 - 1, x2] = 1
      
          # diagonals
          for k in range(0, bh - 3):
              # left diagonal shifted right by 1
              out[y1+k+2, x1 + k + 2] = 1
      
              # right diagonal shifted right by 1
              out[y1+k+1, x1 + k+1+rw] = 1
      
          return out
      
      def task_func(x):
          out = np.zeros_like(x)
      
          for c in np.unique(x):
              if c == 0:
                  continue
              mask = (x == c)
              out[shear_slant_rect_from_bbox(mask).astype(bool)] = c
      
          return out
      
      ```

    - **hengck23** (2026-04-24T23:38:05.810Z, votes: {'canUpvote': True}):
      best pixel-based onnx: 
      It appears to require 14432 MACs + 136996 bytes + 2349 params, yielding cost = 153777 and score = 13.057
      
      best location based onnx
      It appears to require 60938 MACs + 152253 bytes + 341 params, yielding cost = 213532 and score = 12.728

    - **Russell Kirk** (2026-04-25T00:09:16.713Z, votes: {'canUpvote': True}):
      Across the 265 public examples, no two "8-connected" objects share a row  -- if you "overfit" here, you can earn more points.

    - **hengck23** (2026-04-25T00:49:09.223Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      thanks, i get your hint. now at 13.5!
      
      
      It appears to require 7572 MACs + 90173 bytes + 69 params, yielding cost = 97814 and score = 13.509

- **hengck23** (2026-04-21T01:37:58.140Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  task004 solution
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F9aaa4b2cc535c93de6b4f850c6545028%2FSelection_2946.png?generation=1776735438411111&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb6a47d359d80a017b15e1cdb707d968e%2FSelection_2945.png?generation=1776735457692299&alt=media)
  
  ```
  make_task004_onnx_first
  It appears to require 174267 MACs + 673865 bytes + 126 params, yielding 11.349 points.
  
  make_task004_onnx_v2_shortcut
  It appears to require 146187 MACs + 712001 bytes + 136 params, yielding 11.337 points.
  
  make_task004_onnx_v3_gather_maskbank
  It appears to require 145647 MACs + 1055221 bytes + 54140 params, yielding 10.957 points.
  
  make_task004_onnx_v4_parallel_channels
  It appears to require 147087 MACs + 690305 bytes + 124 params, yielding 11.362 points.
  
  make_task004_onnx_v5_erase_col
  It appears to require 138987 MACs + 654273 bytes + 120 params, yielding 11.416 points.
  
  make_task004_onnx_v6_factorized_keep
  It appears to require 138987 MACs + 654769 bytes + 164 params, yielding 11.415 points.
  
  make_task004_onnx_v7_two_stage_reduce
  It appears to require 123867 MACs + 594289 bytes + 164 params, yielding 11.515 points.
  
  make_task004_onnx_v8_no_min
  It appears to require 115767 MACs + 561889 bytes + 164 params, yielding 11.573 points.
  
  make_task004_onnx_v9_no_keep_gate
  It appears to require 107658 MACs + 529444 bytes + 164 params, yielding 11.635 points.
  
  make_task004_onnx_v10_boundary_aware
  It appears to require 116838 MACs + 597448 bytes + 164 params, yielding 11.521 points.
  
  make_task004_onnx_v11_no_valid_mul
  It appears to require 99558 MACs + 497044 bytes + 164 params, yielding 11.701 points.
  
  make_task004_onnx_v12_mul_move
  It appears to require 83628 MACs + 465688 bytes + 164 params, yielding 11.783 points.
  
  make_task004_onnx_v14_direct_corner_masks
  It appears to require 82008 MACs + 459892 bytes + 44 params, yielding 11.797 points.
  ```

- **hengck23** (2026-04-20T22:21:39.200Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  showing graph diagram as feedback helps
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F41acc4c1b1693e50cd34aa23f1ad14ed%2FSelection_2942.png?generation=1776723697529328&alt=media)

- **hengck23** (2026-04-20T21:05:58.373Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  prompt: "you are now very familiar with task001 which is a broadcasting task. can you compute the best theoretical macs, mem and params? use information on web and experiments we performed so far"
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F57f48b31a9b21f70b3144b20b61b713e%2FSelection_2938.png?generation=1776719330530011&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F91680f6a5ba3527284072bc4e75153f5%2FSelection_2939.png?generation=1776719347769745&alt=media)

- **hengck23** (2026-04-21T06:05:59.453Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbe0298604ae238ba144f18b6ec81ea3a%2FSelection_2970.png?generation=1776751557862797&alt=media)

- **hengck23** (2026-04-20T13:55:59.223Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  ```
  # lb 14.82
  def make_task001_onnx_gather_compact_int32_idx():
      def const(name, arr):
          return onh.from_array(np.asarray(arr), name=name)
  
      src_idx = []
      block_idx = []
      for r in range(9):
          for c in range(9):
              src_idx.append((r % 3) * 3 + (c % 3))
              block_idx.append((r // 3) * 3 + (c // 3))
  
      # changed to int32
      src_idx = np.array(src_idx, dtype=np.int32)
      block_idx = np.array(block_idx, dtype=np.int32)
  
      X = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
      Y = oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
  
      init = [
          const("patch_starts", np.array([0, 0], dtype=np.int64)),
          const("patch_ends",   np.array([3, 3], dtype=np.int64)),
          const("patch_axes",   np.array([2, 3], dtype=np.int64)),
          const("patch_steps",  np.array([1, 1], dtype=np.int64)),
  
          const("c0_starts", np.array([0], dtype=np.int64)),
          const("c0_ends",   np.array([1], dtype=np.int64)),
          const("c0_axes",   np.array([1], dtype=np.int64)),
          const("c0_steps",  np.array([1], dtype=np.int64)),
  
          const("one_scalar", np.array(1.0, dtype=np.float32)),
  
          const("shape_patch_flat", np.array([1, 10, 9], dtype=np.int64)),
          const("shape_mask_flat",  np.array([1, 1, 9], dtype=np.int64)),
          const("shape_out9",       np.array([1, 10, 9, 9], dtype=np.int64)),
  
          # int32 here
          const("src_idx", src_idx),
          const("block_idx", block_idx),
  
          const("pads_30", np.array([0, 0, 0, 0, 0, 0, 21, 21], dtype=np.int64)),
          const("pad_value", np.array(0.0, dtype=np.float32)),
      ]
  
      nodes = [
          oh.make_node("Slice",
              inputs=["input", "patch_starts", "patch_ends", "patch_axes", "patch_steps"],
              outputs=["patch"], name="slice_patch"),
          oh.make_node("Slice",
              inputs=["patch", "c0_starts", "c0_ends", "c0_axes", "c0_steps"],
              outputs=["channel0"], name="slice_channel0"),
          oh.make_node("Sub",
              inputs=["one_scalar", "channel0"],
              outputs=["mask"], name="make_mask"),
          oh.make_node("Reshape",
              inputs=["patch", "shape_patch_flat"],
              outputs=["patch_flat"], name="reshape_patch_flat"),
          oh.make_node("Reshape",
              inputs=["mask", "shape_mask_flat"],
              outputs=["mask_flat"], name="reshape_mask_flat"),
          oh.make_node("Gather",
              inputs=["mask_flat", "block_idx"],
              outputs=["trigger81"], axis=2, name="gather_trigger81"),
          oh.make_node("Cast",
              inputs=["trigger81"],
              outputs=["trigger81_bool"],
              to=TensorProto.BOOL,
              name="cast_trigger_bool"),
          oh.make_node("Where",
              inputs=["trigger81_bool", "src_idx", "block_idx"],
              outputs=["gather_idx81"], name="where_choose_index"),
          oh.make_node("Gather",
              inputs=["patch_flat", "gather_idx81"],
              outputs=["out81_raw"], axis=2, name="gather_pixels"),
          oh.make_node("Reshape",
              inputs=["out81_raw", "shape_out9"],
              outputs=["out9"], name="reshape_out9"),
          oh.make_node("Pad",
              inputs=["out9", "pads_30", "pad_value"],
              outputs=["output"],
              mode="constant", name="pad_to_30"),
      ]
  
      graph = oh.make_graph(nodes, "task001_gather_compact_int32_idx", [X], [Y], initializer=init)
      model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)], producer_name="chatgpt")
      model.ir_version = 7
      onnx.checker.check_model(model)
      return model
  ```
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F2bb60629d1067fcc08e5fb313e1096fa%2FSelection_2923.png?generation=1776693343405998&alt=media)

- **hengck23** (2026-04-25T11:17:16.867Z, votes: {'canUpvote': True}):
  i ask gemini3-proto code task004, given a python np code reference. interesting he chooses to use pytorch to code and covertonnx from there.
  
  first try:
  It appears to require 13778 MACs + 216391 bytes + 0 params, yielding cost = 230169 and score = 12.653
  
  ```
  import torch
  import torch.nn.functional as F
  import onnx
  
  import torch
  import torch.nn.functional as F
  import onnx
  
  
  class VectorizedTaskModel(torch.nn.Module):
      def forward(self, onehot):
          # onehot shape: (1, 10, 30, 30), type: float32
          g = onehot[0]
          device = g.device
  
          # 1. Preprocessing: Find H, W
          padding = (g.sum(dim=0) == 0)
  
          H = padding[:, 0].long().argmax()
          # Fallback in case there is no padding at all (image is perfectly 30x30)
          H = torch.where((H == 0) & ~padding[0, 0], torch.tensor(30, dtype=torch.int64, device=device), H)
  
          W = padding[0, :].long().argmax()
          W = torch.where((W == 0) & ~padding[0, 0], torch.tensor(30, dtype=torch.int64, device=device), W)
  
          # Extract label and crop to 16x16
          label = g.argmax(dim=0)
          out16 = label[:16, :16]
  
          # 2. Corner Detection
          cur = out16[1:, 1:]
          left = out16[1:, :-1]
          up = out16[:-1, 1:]
  
          corner_inner = (cur != 0) & (cur == left) & (cur == up)
          corner = F.pad(corner_inner, pad=(1, 0, 1, 0), value=False)
  
          # 3. Vectorized Erasing
          # Erase[y, x0] and [y-1, x0]
          corner_up = F.pad(corner[1:, :], pad=(0, 0, 0, 1), value=False)
          erase_mask = corner | corner_up
          out16 = torch.where(erase_mask, torch.tensor(0, dtype=out16.dtype, device=device), out16)
  
          # 4. Vectorized Redrawing
          # Instead of a 1D vector, use a 2D grid of the corner colors
          # (This prevents overwriting the top row with 0s!)
          c_grid = label[:16, :16] * corner.long()
  
          # Shift the color left, and up-left
          draw_c_left = F.pad(c_grid[:, 1:], pad=(0, 1, 0, 0), value=0)  # Moves `c` to [y, x0-1]
          draw_c_up_left = F.pad(c_grid[1:, 1:], pad=(0, 1, 0, 1), value=0)  # Moves `c` to[y-1, x0-1]
  
          # Shift the boolean masks to match where to draw
          corner_left = F.pad(corner[:, 1:], pad=(0, 1, 0, 0), value=False)
          corner_up_left = F.pad(corner[1:, 1:], pad=(0, 1, 0, 1), value=False)
  
          # Draw the colors into the tensor
          out16 = torch.where(corner_left, draw_c_left, out16)
          out16 = torch.where(corner_up_left, draw_c_up_left, out16)
  
          # 5. Vectorized Transition Pixel Finding (mimicking x1 = np.where(...)[0][0])
          c_per_row = c_grid.max(dim=1, keepdim=True).values  # Shape [16, 1]
          is_zero = (out16[:, :-1] == 0)
          is_c = (out16[:, 1:] == c_per_row)
          valid_row = (c_per_row != 0)
  
          trans = is_zero & is_c & valid_row  # Boolean mask of transitions
  
          # Argmax acts as [0][0] -> gets the index of the FIRST transition
          first_trans_idx = trans.to(torch.int64).argmax(dim=1)
  
          # Reconstruct exactly one boolean True per row at that index
          x_range = torch.arange(15, device=device).unsqueeze(0)
          first_trans_mask = (x_range == first_trans_idx.unsqueeze(1)) & valid_row
  
          # Pad right to realign with the 16x16 grid
          first_trans_padded = F.pad(first_trans_mask, pad=(0, 1, 0, 0), value=False)
          out16 = torch.where(first_trans_padded, c_per_row, out16)
  
          # 6. Global Shift Left (which moves image Right by 1)
          shifted = F.pad(out16[:, :-1], pad=(1, 0, 0, 0), value=0)
  
          # 7. Postprocessing (One-hot & Restore Padding)
          out = F.one_hot(shifted, num_classes=10).float()
          out = out.permute(2, 0, 1)  # Transpose to (10, 16, 16)
  
          y_coords = torch.arange(16, device=device).view(16, 1)
          x_coords = torch.arange(16, device=device).view(1, 16)
  
          mask = (y_coords >= H) | (x_coords >= W)
          out = torch.where(mask.unsqueeze(0), torch.tensor(0.0, device=device), out)
  
          # Final pad back to 30x30 and add batch dimension
          out = F.pad(out, pad=(0, 30 - 16, 0, 30 - 16), value=0.0)
          out = out.unsqueeze(0)
  
          return out
  
  
  # You can export it exactly as before:
  # torch.onnx.export(model, dummy_input, "task_solver.onnx", opset_version=17, ...)
  
  
  def create_onnx_graph(output_path="task_solver.onnx"):
      """Exports the PyTorch model to a static ONNX computation graph."""
      model = VectorizedTaskModel()
      model.eval()
  
      # Dummy input matching the expected shape (1 Batch, 10 Channels, 30 H, 30 W)
      dummy_input = torch.zeros(1, 10, 30, 30, dtype=torch.float32)
  
      print(f"Exporting ONNX graph to {output_path}...")
      torch.onnx.export(
          model,
          dummy_input,
          output_path,
          export_params=True,
          opset_version=17,  # Use opset 17 for robust boolean/padding support
          do_constant_folding=True,
          input_names=['input'],
          output_names=['output'],
      )
  
      # Verify the exported graph
      onnx_model = onnx.load(output_path)
      onnx.checker.check_model(onnx_model)
      print("ONNX graph successfully generated and validated!")
  
  
  if __name__ == "__main__":
      create_onnx_graph(output_path="task004.onnx")
  
  #######################################
  # model = make_task_corner_patch_onnx()
  # onnx.save(model, "task004.onnx")
  
  ```

  - **hengck23** (2026-04-25T11:18:17.433Z, votes: {'canUpvote': True}):
    python np reference
    ```
    
    def detect_corners_label(label):
        cur  = label[1:, 1:]
        left = label[1:, :-1]
        up   = label[:-1, 1:]
    
        corner_inner = (cur != 0) & (cur == left) & (cur == up)
    
        corner = np.zeros_like(label, dtype=bool)
        corner[1:, 1:] = corner_inner
        return corner
    
    def task_onehot_func(onehot):
        g = onehot[0]
        padding = g.sum(0)==0
        H,W = padding[:,0].argmax(),padding[0].argmax() # get first nonzero, that is actual shape without padding
    
        label = onehot[0].argmax(0)
        label = label[:16, :16]
        out16 = label.copy()
    
        corner = detect_corners_label(label)
        ys, xs = np.where(corner)
    
        for y, x0 in zip(ys, xs):
            c = label[y, x0]
    
            # patch right local bottom-right L
            # (simulate moving left)
            out16[y,     x0    ] = 0
            out16[y - 1, x0    ] = 0
            out16[y,     x0 - 1] = c
            out16[y - 1, x0 - 1] = c
    
            # patch right local bottom-left L
            # (simulate moving left)
            x1 = np.where((out16[y,:-1]==0) & (out16[y,1:]==c))[0][0] # only one value
            out16[y, x1] = c
    
        shifted = np.zeros_like(out16)
        shifted[::, 1:] = out16[:,:-1]  #shift right
    
        # convert back to onehot
        out = np.eye(10)[shifted]
        out = out.transpose((2, 0, 1))
    
        # restore padding
        out[0,   H:]=0  #. input and output must have same padding (e,g, check out[0],onehot[0,0][:16,:16])
        out[0,:, W:]=0
    
        # restore to 1x10x30x30
        out = np.pad(out, ((0,0),(0,30-16),(0,30-16)))
        out = out[None,...].astype(np.float32)
        return out
    
    
    ```

- **hengck23** (2026-04-24T12:21:38.890Z, votes: {'canUpvote': True}):
  check these:
  https://www.kaggle.com/code/hengck23/neurogolf-60task-np-solution
  
  prompt " for each of the taskxxx.py, write a corresponding function to make a onnx graph ..."

- **hengck23** (2026-04-24T01:53:57.643Z, votes: {'canUpvote': True}):
  the new chatgpt image2. everything generated automatic
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb7951f5929126c29a89124190b030c24%2FSelection_3061.png?generation=1776995625211176&alt=media)

- **hengck23** (2026-04-23T04:35:19.970Z, votes: {'canUpvote': True}):
  tired of cut and paste, i automate the same task 001 with codex:
  prompt " read program.md and optizmize for 20 steps".
  
  initially, it stops at lb 0.431 after 3 steps. then i say no good and give it some hints. amazing, it get 16.45!
  
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1f48e89d2bb282ad68440262f7e00e73%2FSelection_3034.png?generation=1776918829743937&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F6655f35d7dda84f81ad5d586eb2a6ac1%2FSelection_3033.png?generation=1776918844348944&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff5a4d23115443734a3b47fdba965098d%2FSelection_3035.png?generation=1776919033862705&alt=media)
  ---
  
  conclusion: performance is dependent on the hints you give (i.e. context matter)

- **hengck23** (2026-04-21T14:14:41.657Z, votes: {'canUpvote': True}):
  seems that onehot is not a good representation (when you don't use conv, mlp etc)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5f50bfc7e626852a44f337bdd2798315%2FSelection_2990.png?generation=1776780876234658&alt=media)
  
  graph optimization is a very tiring job. in some cases, 10 compare + concat is better than a onehot opset. in other cases, it is otherwise ...

# kaggle submission error

- Topic ID: 692195
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692195
- Author: @🤞@ (@anglolodorf)
- Posted: 2026-04-16T14:02:46.539712300Z
- Votes: 3
- Total messages: 8

## Body

need to kowns more about kaggle error "Submission Details
hypothese - Version 4

Error · 35m ago

Error processing one or more onnx networks.
Uploaded files
submission.zip![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27319956%2Fd9e9d63dcc20f9334dc9bf244a7471a8%2Fkaggle.png?generation=1776348152660651&alt=media)

(70 K)

## Comments (8)

- **yash bhaskar** (2026-04-18T13:30:21.573Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  It would be great if the error message can include exactly which onnx networks gave the error.

  - **Boredom** (2026-04-20T01:54:11.460Z, votes: {'canUpvote': True}):
    I think so too

- **Michael D. Moffitt** (2026-04-21T23:18:23.767Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  We've just updated the error message to list all failing tasks.  Hope this helps!

- **Patrik** (2026-04-20T02:44:20.923Z, votes: {'canUpvote': True}):
  Hey there—
  
  I agree with the other comments; it would be much more helpful if the error message specified exactly which task triggered the failure.
  
  That said, I was able to narrow this down to the use of GreaterOrEqual and LessOrEqual operations. These were introduced/updated in Opset 12, and it seems the Kaggle scoring environment for this specific competition might be struggling with them.
  
  Once I removed those Opset 12 operations, the error cleared up for me. Hope that helps!

- **belem kalilou** (2026-04-19T11:08:49.670Z, votes: {'canUpvote': True}):
  I have the same error too. please help me

- **Geremie Yeo** (2026-04-19T06:19:48.763Z, votes: {'canUpvote': True}):
  ```
  class Task004(nn.Module):
      """
      Task 004 perfectly aligns sheared voxel/parallelogram blocks mathematically. 
      It identifies maximum-Y constraints natively without iterating grids using explicit
      tensor multiplication yielding correct mask topologies statically evaluating zero loops.
      """
      def forward(self, x):
          B, C, H, W = x.shape
          c_mask = x[:, 1:] 
          
          device = x.device
          y_coords = torch.arange(30, device=device).view(1, 1, 30, 1).expand(B, 9, 30, 30).float()
          x_coords = torch.arange(30, device=device).view(1, 1, 1, 30).expand(B, 9, 30, 30).float()
          
          max_y_per_color = (c_mask * y_coords).amax(dim=(2,3), keepdim=True)
          is_max_r = (y_coords == max_y_per_color) & (c_mask > 0.5)
          
          is_max_rm1 = (y_coords == max_y_per_color - 1) & (c_mask > 0.5)
          max_x_rm1 = (is_max_rm1.float() * x_coords).amax(dim=(2,3), keepdim=True)
          is_that_pixel = is_max_rm1 & (x_coords == max_x_rm1) & (c_mask > 0.5)
          
          do_not_shift = (is_max_r | is_that_pixel).float()
          do_not_shift_any = (do_not_shift.sum(dim=1, keepdim=True) > 0.5).float()
          
          c_any = (c_mask.sum(dim=1, keepdim=True) > 0.5).float()
          shift_mask = c_any * (1.0 - do_not_shift_any)
          
          c_shifted = torch.roll(x, 1, dims=3)
          left_shift_mask = torch.roll(shift_mask, 1, dims=3)
          
          out_shifted = c_shifted * left_shift_mask
          out_stay = x * c_any * (1.0 - shift_mask)
          
          out_1_9 = out_shifted[:, 1:] + out_stay[:, 1:]
          out_0 = 1.0 - out_1_9.sum(dim=1, keepdim=True)
          return torch.cat([out_0, out_1_9], dim=1)
  ```
  
  This is my Task004 - throws an error. This is a wrong solution btw. But I would expect it to get 0 rather than error out

- **габитов айрат айдарович** (2026-04-17T12:00:17.937Z, votes: {'canUpvote': True}):
  so what could be a problem?

- **(unknown)** (2026-04-19T12:58:27.183Z, votes: {}):
  (deleted)

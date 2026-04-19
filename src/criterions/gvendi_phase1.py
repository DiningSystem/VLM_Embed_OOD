import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict
import gc

from prepare_one_model_training import OneModelTrainer

class GvendiPhase1(nn.Module):
    def __init__(self, args, model_trainer):
        super().__init__()
        self.args = args
        self.model_trainer = model_trainer
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
    
    @staticmethod
    def _detach_hook(module, args):
        # Hàm cắt đồ thị tính toán
        hidden_states = args[0].detach().requires_grad_(True)
        return (hidden_states,) + args[1:]


    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    

    @staticmethod
    def _teacher_per_sample_signal(
        teacher_qry_reps: torch.Tensor,
        teacher_pos_reps: torch.Tensor,
    ) -> torch.Tensor:
        return ((teacher_qry_reps - teacher_pos_reps) ** 2).sum(dim=-1)


    def _per_sample_grads(self, sample_loss):
        bs = sample_loss.size(0)
        per_sample_grads = []

        for i in range(bs):
            is_last_sample = (i == bs - 1)
            
            # 1. Gọi backward cho ĐÚNG MẪU THỨ i
            # Chỉ giữ lại đồ thị nếu chưa đến mẫu cuối cùng
            sample_loss[i].backward(retain_graph=not is_last_sample)
            
            grads_i = []

            for p in self.model_trainer.target_params:
                if p.grad is not None:
                    # 2. Tách khỏi đồ thị, về CPU, ép kiểu và làm phẳng (chuẩn nhất)
                    grads_i.append(p.grad.detach().cpu().half().view(-1))
                    
                    # 3. Dọn sạch gradient ngay lập tức (Thay thế hoàn toàn cho model.zero_grad())
                    p.grad = None
            
            # 4. Nối các gradient của các layer lại thành 1 vector cho mẫu thứ i
            if grads_i:
                grads_i_cat = torch.cat(grads_i)
                per_sample_grads.append(grads_i_cat)

        # Xóa tensor loss để PyTorch tự do dọn dẹp phần VRAM còn lại
        del sample_loss
        
        # Gộp tất cả các mẫu lại thành shape: [Batch_size, Tổng_số_chiều_gradient]
        return torch.stack(per_sample_grads, dim=0)



    def forward(self, input_data: Dict[str, Tensor]):
        model = self.model_trainer.model

        input_qry = input_data['qry']
        input_pos = input_data['pos']

        # Gắn pre_hook vào layer đầu tiên trong nhóm N layer
        target_layers = self.model_trainer.target_layers
        first_target_layer = target_layers[0]
        hook_handle = first_target_layer.register_forward_pre_hook(self._detach_hook)

        # Encode query and positive — get full-dim embeddings (unnormalized)
        qry_reps = model.encode_input(input_qry)[0]
        pos_reps = model.encode_input(input_pos)[0]
        
        if self.world_size > 1:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_pos_reps = self._dist_gather_tensor(pos_reps)
        else:
            all_qry_reps = qry_reps
            all_pos_reps = pos_reps
        
        sample_loss = self._teacher_per_sample_signal(all_qry_reps, all_pos_reps)

        sample_grad = self._per_sample_grads(sample_loss)

        print(sample_grad.shape)

        hook_handle.remove() # Tháo hook để không ảnh hưởng luồng khác
        
        gc.collect()
        torch.cuda.empty_cache()
        exit(0)
        return {
            "loss": contrastive_loss, 
            # "extracted_gradients": final_grad_vector
        }

        
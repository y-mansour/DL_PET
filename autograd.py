class fwd_map_torch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        
        # x has dimensions B C H W
        
        x = x.squeeze(0).permute(1,2,0)
        
        #ctx.save_for_backward(x)
        #proj = Map(x_cupy)
        
        x_cupy = cp.from_dlpack(x.detach())  
        Ax = proj.forward(x_cupy)
        torch_Ax = torch.from_dlpack(Ax)

        return torch_Ax

    @staticmethod
    def backward(ctx, grad_output):

        #x, = ctx.saved_tensors
        #x_cupy = cp.from_dlpack(x.detach())
        #proj = Map(x_cupy)
        
        grad_output_cupy = cp.from_dlpack(grad_output)
        grad_cupy = proj.adjoint(grad_output_cupy)
        grad_torch = torch.from_dlpack(grad_cupy) 
        grad_torch = grad_torch.permute(2,0,1).unsqueeze(0)

        return grad_torch   
    
class transp_map_torch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y):
        
        y_cupy = cp.from_dlpack(y.detach()) 
        ATy = proj.adjoint(y_cupy)
        torch_ATy = torch.from_dlpack(ATy)
        torch_ATy = torch_ATy.permute(2,0,1).unsqueeze(0)

        return torch_ATy

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_output = grad_output.squeeze(0).permute(1,2,0)
        grad_output_cupy = cp.from_dlpack(grad_output)
        grad_cupy = proj.forward(grad_output_cupy)
        grad_torch = torch.from_dlpack(grad_cupy)        

        return grad_torch      

A = fwd_map_torch.apply
AT = transp_map_torch.apply
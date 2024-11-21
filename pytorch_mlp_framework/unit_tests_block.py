import torch
import unittest
from model_architectures import ConvolutionalProcessingBlockBN, ConvolutionalDimensionalityReductionBlockBN

class TestBlocks(unittest.TestCase):
 
    def test_convolutional_processing_block_bn(self):
        # Test initialization and forward propagation for ConvolutionalProcessingBlockBN
        input_shape = (8, 3, 32, 32)  # (batch_size, channels, height, width)
        num_filters = 16
        kernel_size = 3
        padding = 1
        bias = False
        dilation = 1
     
        # Create block instance
        processing_block = ConvolutionalProcessingBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation)
     
        # Create a random tensor with input shape
        x = torch.randn(input_shape)
     
        # Perform forward propagation
        out = processing_block(x)
     
        # Check the output shape
        self.assertEqual(out.shape, x.shape, "The output shape does not match the input shape with skip connection")
        print("ConvolutionalProcessingBlockBN test passed.")
 
    def test_convolutional_dimensionality_reduction_block_bn(self):
        # Test initialization and forward propagation for ConvolutionalDimensionalityReductionBlockBN
        input_shape = (8, 3, 32, 32)  # (batch_size, channels, height, width)
        num_filters = 16
        kernel_size = 3
        padding = 1
        bias = False
        dilation = 1
        reduction_factor = 2
     
        # Create block instance
        reduction_block = ConvolutionalDimensionalityReductionBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor)
     
        # Create a random tensor with input shape
        x = torch.randn(input_shape)
     
        # Perform forward propagation
        out = reduction_block(x)
     
        # Expected output shape after avg_pool2d with reduction_factor
        expected_height = input_shape[2] // reduction_factor
        expected_width = input_shape[3] // reduction_factor
        self.assertEqual(out.shape, (input_shape[0], num_filters, expected_height, expected_width), 
                         "The output shape of the reduction block is incorrect.")
        print("ConvolutionalDimensionalityReductionBlockBN test passed.")

if __name__ == '__main__':
    unittest.main()

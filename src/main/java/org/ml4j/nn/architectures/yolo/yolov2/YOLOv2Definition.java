package org.ml4j.nn.architectures.yolo.yolov2;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

public class YOLOv2Definition implements Component3Dto3DGraphDefinition {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	public static final ActivationFunctionType LEAKY_RELU_ACTIVATION_FUNCTION_TYPE 
		= ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LEAKYRELU);

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(608, 608, 3, false);
	}
	
	@Override
	public Neurons3D getOutputNeurons() {
		return new Neurons3D(19, 19, 425, false);
	}

	@Override
	public <T extends NeuralComponent> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		NeuralComponentType<T> spaceToDepthComponentType
			= NeuralComponentType.createSubType(NeuralComponentBaseType.AXONS, "SPACE_TO_DEPTH");
		
		// input_1
		return start
				// conv2d_1
				.withConvolutionalAxons("conv2d_1")
				.withFilterSize(3, 3)
				.withFilterCount(32)
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				// batch_normalization_1
				.withBatchNormAxons("batch_normalization_1")
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				.withActivationFunction("leaky_relu_1", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// max_pooling2d_1
				.withMaxPoolingAxons("max_pooling_1")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(304, 304, 32, false))
				// conv2d_2
				.withConvolutionalAxons("conv2d_2")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				// batch_normalization_2
				.withBatchNormAxons("batch_normalization_2")
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				.withActivationFunction("leaky_relu_2", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// max_pooling2d_2
				.withMaxPoolingAxons("max_pooling_2")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// conv2d_3
				.withConvolutionalAxons("conv2d_3")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_3
				.withBatchNormAxons("batch_normalization_3")
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction("leaky_relu_3", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_4
				.withConvolutionalAxons("conv2d_4")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// batch_normalization_4
				.withBatchNormAxons("batch_normalization_4")
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				.withActivationFunction("leaky_relu_4", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_5
				.withConvolutionalAxons("conv2d_5")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_5
				.withBatchNormAxons("batch_normalization_5")
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction("relu_5", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				.withMaxPoolingAxons("max_pooling_3")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// conv2d_6
				.withConvolutionalAxons("conv2d_6")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_6
				.withBatchNormAxons("batch_normalization_6")
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction("leaky_relu_6", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_7
				.withConvolutionalAxons("conv2d_7")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// batch_normalization_7
				.withBatchNormAxons("batch_normalization_7")
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				.withActivationFunction("leaky_relu_7", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_8
				.withConvolutionalAxons("conv2d_8")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_8
				.withBatchNormAxons("batch_normalization_8")
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction("leaky_relu_8", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				.withMaxPoolingAxons("max_pooling_4")
				.withFilterSize(2, 2)
				.withStride(2, 2)
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// conv2d_9
				.withConvolutionalAxons("conv2d_9")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_9
				.withBatchNormAxons("batch_normalization_9")
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction("leaky_relu_9", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_10
				.withConvolutionalAxons("conv2d_10")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_10
				.withBatchNormAxons("batch_normalization_10")
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction("leaky_relu_10", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_11
				.withConvolutionalAxons("conv2d_11")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_11
				.withBatchNormAxons("batch_normalization_11")
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction("leaky_relu_11", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_12
				.withConvolutionalAxons("conv2d_12")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_12
				.withBatchNormAxons("batch_normalization_12")
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction("leaky_relu_12", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_13
				.withConvolutionalAxons("conv2d_13")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withParallelPaths()
				.withPath()
					// batch_normalization_13
					.withBatchNormAxons("batch_normalization_13")
					.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
					// leaky_re_lu_14
					.withActivationFunction("leaky_relu_13", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					.withMaxPoolingAxons("max_pooling_5")
					.withFilterSize(2, 2)
					.withStride(2, 2)
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// conv2d_14
					.withConvolutionalAxons("conv2d_14")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_14
					.withBatchNormAxons("batch_normalization_14")
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_relu_14", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_15
					.withConvolutionalAxons("conv2d_15")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_15
					.withBatchNormAxons("batch_normalization_15")
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction("leaky_relu_15", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_16
					.withConvolutionalAxons("conv2d_16")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_16
					.withBatchNormAxons("batch_normalization_16")
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_relu_16", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_17
					.withConvolutionalAxons("conv2d_17")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_17
					.withBatchNormAxons("batch_normalization_17")
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction("leaky_relu_17", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_18
					.withConvolutionalAxons("conv2d_18")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_18
					.withBatchNormAxons("batch_normalization_18")
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction("leaky_relu_18", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_19
					.withConvolutionalAxons("conv2d_19")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_19
					.withBatchNormAxons("batch_normalization_19")
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_19
					.withActivationFunction("leaky_relu_19", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// conv2d_20
					.withConvolutionalAxons("conv2d_20")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_20
					.withBatchNormAxons("batch_normalization_20")
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_20
					.withActivationFunction("leaky_relu_20", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				.endPath()
				.withPath()
					// conv2d_21
					.withConvolutionalAxons("conv2d_21")
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// batch_normalization_21
					.withBatchNormAxons("batch_normalization_21")
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// leaky_re_lu_21
					.withActivationFunction("leaky_relu_21", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
					// space_to_depth_x2
					.with3DComponent(neuralComponentFactory.createComponent("space_to_depth_x2", new Neurons3D(38, 38, 64, false), 
							new Neurons3D(19, 19, 256, false), spaceToDepthComponentType), new Neurons3D(19, 19, 256, false))
				.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				// conv2d_22
				.withConvolutionalAxons("conv2d_22")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
				// batch_normalization_22
				.withBatchNormAxons("batch_normalization_22")
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
				// leaky_re_lu_20
				.withActivationFunction("leaky_relu_22", LEAKY_RELU_ACTIVATION_FUNCTION_TYPE, new ActivationFunctionProperties())
				// conv2d_23
				.withConvolutionalAxons("conv2d_23")
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 425, false));
	}
	
	@Override
	public String getName() {
		return "yolo_v2_graph";
	}

}

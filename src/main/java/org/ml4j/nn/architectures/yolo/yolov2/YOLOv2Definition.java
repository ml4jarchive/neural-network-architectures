package org.ml4j.nn.architectures.yolo.yolov2;

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
	
	private static final String LEAKY_RELU_CUSTOM_BASE_TYPE_ID = "LEAKY_RELU";
	
	public static final ActivationFunctionType LEAKY_RELU_ACTIVATION_FUNCTION_TYPE 
		= ActivationFunctionType.createCustomBaseType(LEAKY_RELU_CUSTOM_BASE_TYPE_ID);

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
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				// batch_normalization_1
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(608, 608, 32, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// max_pooling2d_1
				.withMaxPoolingAxons()
				.withConnectionToNeurons(new Neurons3D(304, 304, 32, false))
				// conv2d_2
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				// batch_normalization_2
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(304, 304, 64, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// max_pooling2d_2
				.withMaxPoolingAxons()
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// conv2d_3
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_3
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_4
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				// batch_normalization_4
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(152, 152, 64, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_5
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				// batch_normalization_5
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(152, 152, 128, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				.withMaxPoolingAxons()
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// conv2d_6
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_6
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_7
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				// batch_normalization_7
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(76, 76, 128, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_8
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				// batch_normalization_8
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(76, 76, 256, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				.withMaxPoolingAxons()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// conv2d_9
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_9
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_10
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_10
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_11
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				// batch_normalization_11
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_12
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				// batch_normalization_12
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(38, 38, 256, false))
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_13
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
				.withParallelPaths()
				.withPath()
					// batch_normalization_13
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(38, 38, 512, false))
					// leaky_re_lu_14
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					.withMaxPoolingAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// conv2d_14
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_14
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_15
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_15
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_16
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_16
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_17
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					// batch_normalization_17
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 512, false))
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_18
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_18
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_19
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_19
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_19
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// conv2d_20
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// batch_normalization_20
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
					// leaky_re_lu_20
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				.endPath()
				.withPath()
					// conv2d_21
					.withConvolutionalAxons()
					.withSamePadding()
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// batch_normalization_21
					.withBatchNormAxons()
					.withConnectionToNeurons(new Neurons3D(38, 38, 64, false))
					// leaky_re_lu_21
					.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
					// space_to_depth_x2
					.with3DComponent(neuralComponentFactory.createComponent(new Neurons3D(38, 38, 64, false), 
							new Neurons3D(19, 19, 256, false), spaceToDepthComponentType), new Neurons3D(19, 19, 256, false))
				.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				// conv2d_22
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
				// batch_normalization_22
				.withBatchNormAxons()
				.withConnectionToNeurons(new Neurons3D(19, 19, 1024, false))
				// leaky_re_lu_20
				.withActivationFunction(LEAKY_RELU_ACTIVATION_FUNCTION_TYPE)
				// conv2d_22
				.withConvolutionalAxons()
				.withSamePadding()
				.withConnectionToNeurons(new Neurons3D(19, 19, 425, false));
	}

}

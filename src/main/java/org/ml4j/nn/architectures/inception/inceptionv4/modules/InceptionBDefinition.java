/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.architectures.inception.inceptionv4.modules;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.architectures.inception.InceptionModuleDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionBDefinition implements InceptionModuleDefinition {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private InceptionV4WeightsLoader weightsLoader;
	private int inceptionBModuleIndex;
	private float regularisationLambda;
	private float batchNormRegularisationLambda;
	private boolean withFreezeOut;

	public InceptionBDefinition(InceptionV4WeightsLoader weightsLoader, int inceptionBModuleIndex) {
		this.inceptionBModuleIndex = inceptionBModuleIndex;
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(17, 17, 1024, false);
	}

	@Override
	public <T extends NeuralComponent> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		int initialComponentIndex = inceptionBModuleIndex * 10 + 44;
		return start
				.withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + initialComponentIndex + "_kernel0", 1, 1, 1024, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_beta0", 384))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_moving_mean0", 384))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_moving_variance0", 384))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 1) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 2) + "_kernel0", 7, 1, 192, 224))
				.withFilterSize(7, 1).withFilterCount(224).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_beta0", 224))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_mean0", 224))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_variance0", 224))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 3) + "_kernel0", 1, 7, 224, 256))
				.withFilterSize(1, 7).withFilterCount(224).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 4) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 5) + "_kernel0", 1, 7, 192, 192))
				.withFilterSize(1, 7).withFilterCount(192).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 6) + "_kernel0", 7, 1, 192, 224))
				.withFilterSize(7, 1).withFilterCount(224).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_beta0", 224))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_mean0", 224))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_variance0", 224))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 7) + "_kernel0", 1, 7, 224, 224))
				.withFilterSize(1, 7).withFilterCount(224).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 7) + "_beta0", 224))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 7) + "_moving_mean0", 224))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 7) + "_moving_variance0", 224))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 8) + "_kernel0", 7, 1, 224, 256))
				.withFilterSize(7, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 8) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 8) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 8) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withAveragePoolingAxons().withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(17, 17, 1024, false)).withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 9) + "_kernel0", 1, 1, 1024, 128))
				.withFilterSize(1, 1).withFilterCount(128).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 128, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 9) + "_beta0", 128))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 9) + "_moving_mean0", 128))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 9) + "_moving_variance0", 128))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 128, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

}

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
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionADefinition implements InceptionModuleDefinition {

	private InceptionV4WeightsLoader weightsLoader;
	private int inceptionAModuleIndex;
	private float regularisationLambda;
	private float batchNormRegularisationLambda;
	private boolean withFreezeOut;

	public InceptionADefinition(InceptionV4WeightsLoader weightsLoader, int inceptionAModuleIndex) {
		this.inceptionAModuleIndex = inceptionAModuleIndex;
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(35, 35, 384, false);
	}

	@Override
	public <T extends NeuralComponent> Components3DGraphBuilder<?, ?, T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start) {
		
		int initialComponentIndex = inceptionAModuleIndex * 7 + 12;
		return start
				.withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + initialComponentIndex + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + initialComponentIndex + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 1) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_beta0", 64))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_variance0", 64))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 2) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 3) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_beta0", 64))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_variance0", 64))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 4) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 5) + "_kernel0", 3, 3, 96, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()

				.withAveragePoolingAxons().withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(35, 35, 384, false)).withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 6) + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
		
	}

}

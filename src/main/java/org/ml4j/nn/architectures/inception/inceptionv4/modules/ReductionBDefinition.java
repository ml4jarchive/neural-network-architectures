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
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class ReductionBDefinition implements Component3Dto3DGraphDefinition {

	private InceptionV4WeightsLoader weightsLoader;
	private boolean withFreezeOut;
	private float regularisationLambda;
	private float batchNormRegularisationLambda;

	public ReductionBDefinition(InceptionV4WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
	}
	
	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(17, 17, 1024, false);
	}

	@Override
	public Neurons3D getOutputNeurons() {
		return new Neurons3D(8, 8, 1536, false);
	}

	public <T extends NeuralComponent> Components3DGraphBuilder<?, ?, T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start) {
		return start
				.withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (114) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (114) + "_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (114) + "_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (114) + "_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (115) + "_kernel0", 3, 3, 192, 192))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(192).withValidPadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (115) + "_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (115) + "_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (115) + "_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 192, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (116) + "_kernel0", 1, 1, 1024, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (116) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (116) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (116) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (117) + "_kernel0", 7, 1, 256, 256))
				.withFilterSize(7, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (117) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (117) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (117) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (118) + "_kernel0", 1, 7, 256, 320))
				.withFilterSize(1, 7).withFilterCount(320).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 320, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (118) + "_beta0", 320))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (118) + "_moving_mean0", 320))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (118) + "_moving_variance0", 320))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 320, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU))
				.withConvolutionalAxons()
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (119) + "_kernel0", 3, 3, 320, 320))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(320).withValidPadding()
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 320, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (119) + "_beta0", 320))
				.withMean(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (119) + "_moving_mean0", 320))
				.withVariance(weightsLoader.getBatchNormLayerWeights(
						"batch_normalization_" + (119) + "_moving_variance0", 320))
				.withAxonsContextConfigurer(
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 320, false))
				.withActivationFunction(ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU)).endPath().withPath()
				.withMaxPoolingAxons().withFilterSize(3, 3).withStride(2, 2).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(8, 8, 1024, false)).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.getBaseType(NeuralComponentBaseType.DEFINITION);
	}
}

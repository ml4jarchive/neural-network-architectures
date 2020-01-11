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

import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionV4TailDefinition implements Component3DtoNon3DGraphDefinition {

	private InceptionV4WeightsLoader weightsLoader;
	private float regularisationLambda;
	private DifferentiableActivationFunctionFactory activationFunctionFactory;

	public InceptionV4TailDefinition(InceptionV4WeightsLoader weightsLoader,
			DifferentiableActivationFunctionFactory activationFunctionFactory) {
		this.weightsLoader = weightsLoader;
		this.activationFunctionFactory = activationFunctionFactory;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(8, 8, 1536, false);
	}

	public <T extends NeuralComponent> ComponentsGraphBuilder<?, T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start) {
		return start.withSynapses().withAveragePoolingAxons().withStride(1, 1).withFilterSize(8, 8).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(1, 1, 1536, false)).endSynapses().withFullyConnectedAxons()
				.withConnectionWeights(weightsLoader.getDenseLayerWeights("dense_1_kernel0", 1001, 1536))
				.withBiases(weightsLoader.getDenseLayerWeights("dense_1_bias0", 1001, 1))
				.withAxonsContextConfigurer(c -> c.withRegularisationLambda(regularisationLambda)).withBiasUnit()
				.withConnectionToNeurons(new Neurons(1001, false))
				.withActivationFunction(activationFunctionFactory.createSoftmaxActivationFunction());
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.DEFINITION;
	}
}

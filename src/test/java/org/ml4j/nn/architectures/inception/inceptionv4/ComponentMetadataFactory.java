package org.ml4j.nn.architectures.inception.inceptionv4;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * An example of using a custom NeuralComponentFactory to create cusom NeuralComponents into InceptionV4 graphs.
 * 
 * Such custom components can be created in InceptionV4 graphs by a InceptionV4Definition in the same way as the default components
 * 
 * @author Michael Lavelle
 */
public class ComponentMetadataFactory implements NeuralComponentFactory<ComponentMetadata>{

	@Override
	public ComponentMetadata createAveragePoolingAxonsComponent(Neurons3D arg0, Neurons3D arg1, Axons3DConfig arg2) {
		return new ComponentMetadata(arg0, arg1, "Average Pooling Axons");
	}

	@Override
	public <N extends Neurons> ComponentMetadata createBatchNormAxonsComponent(N arg0, N arg1) {
		return new ComponentMetadata(arg0, arg1, "Batch Norm Axons");
	}

	@Override
	public <N extends Neurons> ComponentMetadata createBatchNormAxonsComponent(N arg0, N arg1, Matrix arg2, Matrix arg3,
			Matrix arg4, Matrix arg5) {
		return new ComponentMetadata(arg0, arg1, "Batch Norm Axons");
	}

	@Override
	public ComponentMetadata createConvolutionalAxonsComponent(Neurons3D arg0, Neurons3D arg1, Axons3DConfig arg2,
			Matrix arg3, Matrix arg4) {
		return new ComponentMetadata(arg0, arg1, "Convolutional Axons");
	}

	@Override
	public ComponentMetadata createConvolutionalBatchNormAxonsComponent(Neurons3D arg0, Neurons3D arg1) {
		return new ComponentMetadata(arg0, arg1, "Convolutional Batch Norm Axons");
	}

	@Override
	public ComponentMetadata createConvolutionalBatchNormAxonsComponent(Neurons3D arg0, Neurons3D arg1, Matrix arg2,
			Matrix arg3, Matrix arg4, Matrix arg5) {
		return new ComponentMetadata(arg0, arg1 ,"Convolutional Batch Norm Axons");
	}

	@Override
	public ComponentMetadata createDifferentiableActivationFunctionComponent(Neurons arg0,
			DifferentiableActivationFunction arg1) {
		return new ComponentMetadata(arg0, arg0, "Activation Function:" + arg1.getClass());
	}
	
	@Override
	public ComponentMetadata createDifferentiableActivationFunctionComponent(Neurons arg0,
			ActivationFunctionType arg1) {
		return new ComponentMetadata(arg0, arg0, "Activation Function:" + arg1.getQualifiedId());
	}

	@Override
	public ComponentMetadata createDirectedComponentBipoleGraph(Neurons arg0, Neurons arg1,
			List<ComponentMetadata> arg2, PathCombinationStrategy arg3) {
		return new ComponentMetadata(arg0, arg1, "Bipole Graph with strategy:" + arg3);
	}

	@Override
	public ComponentMetadata createDirectedComponentChain(List<ComponentMetadata> arg0) {
		return new ComponentMetadata(arg0.get(0).getInputNeurons(), arg0.get(arg0.size() -1).getOutputNeurons(), "Component Chain with " + arg0.size() + " components");
	}

	@Override
	public ComponentMetadata createFullyConnectedAxonsComponent(Neurons arg0, Neurons arg1, Matrix arg2, Matrix arg3) {
		return new ComponentMetadata(arg0, arg1, "Fully Connected Axons Component");
	}

	@Override
	public ComponentMetadata createMaxPoolingAxonsComponent(Neurons3D arg0, Neurons3D arg1, Axons3DConfig arg2,
			boolean arg3) {
		return new ComponentMetadata(arg0, arg1, "Max Pooling Axons Component");
	}

	@Override
	public <N extends Neurons> ComponentMetadata createPassThroughAxonsComponent(N arg0, N arg1) {
		return new ComponentMetadata(arg0, arg1, "Pass through Axons Component");
	}

}

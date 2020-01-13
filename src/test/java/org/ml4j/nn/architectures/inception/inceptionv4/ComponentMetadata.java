package org.ml4j.nn.architectures.inception.inceptionv4;

import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.neurons.Neurons;

/**
 * An example of using a custom NeuralComponent as an alternative to one of the default ml4j NeuralComponents 
 * such as DefaultChainableDirectedComponent.
 * 
 * Such custom components can be created in InceptionV4 graphs by a InceptionV4Definition in the same way as the default components
 * 
 * @author Michael Lavelle
 */
public class ComponentMetadata implements NeuralComponent {
	
	private Neurons inputNeurons;
	private Neurons outputNeurons;

	private String description;
	
	public ComponentMetadata(Neurons inputNeurons, Neurons outputNeurons, String description) {
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
		this.description = description;
	}

	@Override
	public NeuralComponentType<NeuralComponent> getComponentType() {
		return NeuralComponentType.createCustomBaseType(ComponentMetadata.class.getName());
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}
	
	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

	public String getDescription() {
		return description;
	}

	@Override
	public String toString() {
		return "ComponentMetadata [inputNeurons=" + inputNeurons + ", outputNeurons=" + outputNeurons + ", description="
				+ description + "]";
	}
}

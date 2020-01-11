package org.ml4j.nn.architectures.inception.inceptionv4;

import java.util.List;

import org.junit.Assert;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integration test with some mock components and some real components.
 * 
 * Uses a custom NeuralComponentFactory to a graph of ComponentMetadata components, in a graph that matches the type of graph 
 * created by a real NeuralComponentFactory (eg. a DirectedComponentFactory instance).
 * 
 * Showcases how a InceptionV4Definition can be used to perform other behaviour other than network creation, such
 * as metadata extraction, or to generate networks in alternative programming languages or alternative frameworks.
 * 
 * @author Michael Lavelle
 */
public class InceptionV4DefinitionMetadataExtractionTest extends InceptionV4DefinitionTestBase<ComponentMetadata> {

	private static final Logger LOGGER = LoggerFactory.getLogger(InceptionV4DefinitionMetadataExtractionTest.class);
	
	@Override
	protected NeuralComponentFactory<ComponentMetadata> createNeuralComponentFactory() {
		return new ComponentMetadataFactory();
	}

	@Override
	protected void runAssertionsOnCreatedComponentGraph(
			ComponentsGraphBuilder<?, ComponentMetadata> componentGraph) {
		
		List<ComponentMetadata> sequentialComponents = componentGraph.getComponents();
		
		Assert.assertEquals(31, sequentialComponents.size());
		
		for (ComponentMetadata component : sequentialComponents) {
			LOGGER.debug(component.toString());
		}
	}
}

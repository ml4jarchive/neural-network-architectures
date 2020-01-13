package org.ml4j.nn.architectures.inception.inceptionv4;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.sessions.Session;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class InceptionV4DefinitionTestBase<T extends NeuralComponent> {
	
	@Mock
	private InceptionV4WeightsLoader mockInceptionV4WeightsLoader;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	private AxonsContext mockAxonsContext;
		
	protected NeuralComponentFactory<T> neuralComponentFactory;
	
	protected abstract NeuralComponentFactory<T> createNeuralComponentFactory();
	
	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.neuralComponentFactory = createNeuralComponentFactory();
		Mockito.when(mockDirectedComponentsContext.getContext(Mockito.any(), Mockito.any())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withRegularisationLambda(Mockito.anyFloat())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withFreezeOut(Mockito.anyBoolean())).thenReturn(mockAxonsContext);
	}
	
	protected abstract void runAssertionsOnCreatedComponentGraph(InceptionV4Definition inceptionV4Definition, 
			InitialComponentsGraphBuilder<T>  componentGraph);

	@Test
	public void testComponentGraphCreation() {
	
		// Start new session, given the component factory and the runtime context.
		Session<T> session = new SessionImpl<>(neuralComponentFactory, mockDirectedComponentsContext);
		
		// Create the InceptionV4Definition
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition( 
				mockInceptionV4WeightsLoader);
		
		// Build a component graph, given this Session and the InceptionV4Definition.
		InitialComponentsGraphBuilder<T> componentGraph = session.startWith(inceptionV4Definition);
			
		// Assert that we now have a component graph.
		Assert.assertNotNull(componentGraph);
		
		// Run additional assertions
		runAssertionsOnCreatedComponentGraph(inceptionV4Definition, componentGraph);
	}

}

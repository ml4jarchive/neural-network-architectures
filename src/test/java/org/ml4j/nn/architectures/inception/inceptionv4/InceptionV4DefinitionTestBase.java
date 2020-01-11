package org.ml4j.nn.architectures.inception.inceptionv4;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.factories.DummyDifferentiableActivationFunctionFactory;
import org.ml4j.nn.neurons.Neurons3D;
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
	
	private DifferentiableActivationFunctionFactory dummyDifferentiableActivationFunctionFactory;
	
	protected NeuralComponentFactory<T> neuralComponentFactory;
	
	protected abstract NeuralComponentFactory<T> createNeuralComponentFactory();
	
	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.dummyDifferentiableActivationFunctionFactory = new DummyDifferentiableActivationFunctionFactory();
		this.neuralComponentFactory = createNeuralComponentFactory();
		Mockito.when(mockDirectedComponentsContext.getContext(Mockito.any(), Mockito.any())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withRegularisationLambda(Mockito.anyFloat())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withFreezeOut(Mockito.anyBoolean())).thenReturn(mockAxonsContext);
	}
	
	private InitialComponents3DGraphBuilder<T> createGraphBuilder(Neurons3D initialNeurons) {
		return new InitialComponents3DGraphBuilderImpl<>(neuralComponentFactory, mockDirectedComponentsContext, initialNeurons) ;
	}
	
	protected abstract void runAssertionsOnCreatedComponentGraph(ComponentsGraphBuilder<?, T>  componentGraph);

	@Test
	public void testComponentGraphCreation() {
		
		// Create the InceptionV4Definition
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition(dummyDifferentiableActivationFunctionFactory, 
				mockInceptionV4WeightsLoader);
				
		// Buiilder a component graph, given this InceptionV4Definition and the components factories.
		ComponentsGraphBuilder<?, T> componentGraph 
				= inceptionV4Definition.createComponentGraph(createGraphBuilder(inceptionV4Definition.getInputNeurons()));
		
		Assert.assertNotNull(componentGraph);
		
		runAssertionsOnCreatedComponentGraph(componentGraph);
	}

}

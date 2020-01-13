package org.ml4j.nn.architectures.yolo.yolov2;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.sessions.Session;
import org.ml4j.nn.sessions.SessionImpl;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class YOLOv2DefinitionTestBase<T extends NeuralComponent> {
	
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
	
	protected abstract void runAssertionsOnCreatedComponentGraph(YOLOv2Definition yOLOv2Definition, 
			InitialComponents3DGraphBuilder<T>  componentGraph);

	@Test
	public void testComponentGraphCreation() {
	
		// Start new session, given the component factory and the runtime context.
		Session<T> session = new SessionImpl<>(neuralComponentFactory, mockDirectedComponentsContext);
		
		// Create the YOLOv2Definition
		YOLOv2Definition yoloV2Definition = new YOLOv2Definition();
		
		// Build a component graph, given this Session and the YOLOv2Definition.
		InitialComponents3DGraphBuilder<T> componentGraph = session.startWith(yoloV2Definition);
			
		// Assert that we now have a component graph.
		Assert.assertNotNull(componentGraph);
		
		// Run additional assertions
		runAssertionsOnCreatedComponentGraph(yoloV2Definition, componentGraph);
	}

}

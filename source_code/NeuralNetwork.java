//Author: Cosmos Bennett

public class NeuralNetwork
{
    //attribs
    private double eta = 0.2;
    private double alpha = 0.5;
    private Layers layers = new Layers ( );
    private Architecture architecture = new Architecture ( "2,2,1");
    
    // constructor
    public NeuralNetwork ( )
    {
        //define neurons
        for ( int lSI = 0; lSI < architecture.size ( ); lSI ++ )
        {
            layers.add ( new Layer ( ) );
            
            for ( int lI = 0; lI <= architecture.get ( lSI ); lI ++ )
            {
                int numberOfWeightsFromNextNeuron = ( lSI + 1 ) < architecture.size ( ) ? architecture.get ( lSI + 1 ): 0;
                Neuron newNeuron = new Neuron ( eta, alpha, lI, numberOfWeightsFromNextNeuron );
                
                layers.get ( lSI ).add ( newNeuron );
            }
            
            layers.get ( lSI ).get ( layers.get ( lSI ).size ( ) - 1 ).setOutcome ( 1.0 );
        }
    }
    
    //methods
    public void doForwardPropagation ( int [ ] inputs )
    {
        for ( int iI = 0; iI < inputs.length; iI ++ )
            layers.get ( 0 ).get ( iI ).setOutcome ( inputs [ iI ] );
            
        for ( int lSI = 1; lSI < architecture.size ( ); lSI ++ )
        {
            for ( int lI = 0; lI < architecture.get ( lSI ); lI ++ )
            {
                Layer priorLayer = layers.get ( lSI - 1 );
                layers.get ( lSI ).get ( lI ).doForwardPropagation ( priorLayer );
            }
        }
    }

    public void doBackwardPropagation ( int target )
    {
        //output grad
        Neuron outputNeuron = layers.get ( layers.size ( ) - 1 ).get ( 0 );
        outputNeuron.setOutcomeGradient ( target );
        
        //hidden grad
        for ( int lSI = architecture.size ( ) - 2; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer nextLayer = layers.get ( lSI + 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ); lI ++ )
                currentLayer.get ( lI ).setHiddenGradient ( nextLayer );
        }
        
        //weight update
        for ( int lSI = architecture.size ( ) - 1; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer priorLayer = layers.get ( lSI - 1 );
            
            
            for ( int lI = 0; lI < currentLayer.size ( ) - 1; lI ++ )
                currentLayer.get ( lI ).updateWeights ( priorLayer );
        }
    }
    
    public double getOutcome ( )
    {
        return layers.get ( architecture.size ( ) - 1 ).get ( 0 ).getOutcome ( );
    }
}

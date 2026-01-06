//Author: Cosmos Bennett

import java.util.ArrayList;
import java.util.Random;

public class Neuron
{
    //attribs
    private double eta;
    private double alpha;
    private double outcome;
    private double gradient;
    private int numberOfWeightsFromNextNeuron;
    private int neuronId;
    private ArrayList <Synapse> weights;
    
    public Neuron ( double eta, double alpha, int neuronId, int numberOfWeightsFromNextNeuron )
    {
        this.gradient = 0.0;
        this.eta = eta;
        this.alpha = alpha;
        this.numberOfWeightsFromNextNeuron = numberOfWeightsFromNextNeuron;
        this.neuronId = neuronId;
        
        weights = new ArrayList <Synapse> ( );
        for ( int wI = 0; wI < numberOfWeightsFromNextNeuron; wI ++ )
        {
            weights.add ( new Synapse ( ) );
            weights.get ( wI ).setWeight ( new Random ( ).nextDouble ( ) );
        }
    }
    
    public double getOutcome ( )
    {
        return outcome;
    }
    
    public double getGradient ( )
    {
        return gradient;
    }
    
    public ArrayList <Synapse> getWeights ( )
    {
        return weights;
    }
    
    public double getActivation ( double value )
    {
        //tanh ( x ) ?
        return Math.tanh ( value );
    }
   
    public double getPrimeActivation ( double value )
    {
        //1 - tanh ( x ) ^ 2?
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }
    
    public double getDistributedWeightSigma ( Layer nextLayer )
    {
        double sigma = 0;
        
        for ( int nLI = 0; nLI < nextLayer.size ( ) - 1; nLI ++ )
            sigma += getWeights ( ).get ( nLI ).getWeight ( ) * nextLayer.get ( nLI ).getGradient ( );
            
        return sigma;
    }
    
    public void setHiddenGradient ( Layer nextLayer )
    {
        double delta = getDistributedWeightSigma ( nextLayer );
        
        setGradient ( delta * getPrimeActivation ( outcome ) );
    }
    
    public void setOutcomeGradient ( double target )
    {
        double delta = target - outcome;
        
        setGradient ( delta * getPrimeActivation ( outcome ) );
    }
    
    public void setOutcome ( double value )
    {
        outcome = value;
    }
    
    public void setGradient ( double value )
    {
        gradient = value;
    }
    
    
    public void doForwardPropagation ( Layer priorLayer )
    {
        double sigma = 0;
               
        //?? (eta * thisGrad ) + (alpha * priorDeltaWeight)
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
        {
            double priorWeight = priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( );
            double priorOutcome = priorLayer.get ( pLI ).getOutcome ( );
            
            sigma += priorWeight * priorOutcome;
        }
        
        setOutcome ( getActivation ( sigma ) );
    }
    
    public void updateWeights ( Layer priorLayer )
    {
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
        {
            double priorDeltaWeight = priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getDeltaWeight ( );
            double newDeltaWeight = ( eta * getGradient ( ) * priorLayer.get ( pLI ).getOutcome ( ) ) + ( alpha * priorDeltaWeight );
            
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setDeltaWeight ( newDeltaWeight );
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setWeight ( priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) + newDeltaWeight );
        }
    }
}
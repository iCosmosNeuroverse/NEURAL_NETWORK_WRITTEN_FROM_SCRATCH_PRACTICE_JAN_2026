//Author: Cosmos Bennett

import java.util.ArrayList;

public class Architecture extends ArrayList <Integer>
{
    //attrib
    private String description;
    
    //constructor
    public Architecture ( String description )
    {
        String [ ] components = description.split ( "," );
        
        for ( int cI = 0; cI < components.length; cI ++ )
            add ( Integer.parseInt ( components [ cI ] ) );
    }
}
#version 330 core

uniform vec3 ka;            // Ambient reflectivity
uniform vec3 kd;            // Diffuse reflectivity
uniform vec3 ks;            // Specular reflectivity
uniform float shininess;    // Specular shininess factor

uniform vec4 lightPosition;
uniform vec3 lightIntensity;

uniform float lineWidth;
uniform vec4 lineColor;
uniform float showLine;
uniform float showDiff;
uniform sampler2D albedo;

in WireframeVertex {
    vec3 position;
    vec3 normal;
    vec3 color;
    vec2 uv;
    noperspective vec4 edgeA;
    noperspective vec4 edgeB;
    flat int configuration;
} fs_in;

out vec4 fragColor;

vec3 adsModel( const in vec3 pos, const in vec3 n )
{
    // Calculate the vector from the light to the fragment
    vec3 s = normalize( vec3( lightPosition ) - pos );

    // Calculate the vector from the fragment to the eye position (the
    // origin since this is in "eye" or "camera" space
    vec3 v = normalize( -pos );

    // Refleft the light beam using the normal at this fragment
    vec3 r = reflect( -s, n );

    // Calculate the diffuse component
    vec3 diffuse = vec3( max( dot( s, n ), 0.0 ) );

    // Calculate the specular component
    vec3 specular = vec3( pow( max( dot( r, v ), 0.0 ), shininess ) );

    // Combine the ambient, diffuse and specular contributions
    //diffuse *= kd;

    if (showLine == 2.0)
    {
        diffuse = diffuse * texture(albedo, fs_in.uv).rgb;
    }
       
    return lightIntensity * ( ka + diffuse + vec3(0,0,0) * specular );
}

vec4 shadeLine( const in vec4 color )
{
    // Find the smallest distance between the fragment and a triangle edge
    float d;
    if ( fs_in.configuration == 0 )
    {
        // Common configuration
        d = min( fs_in.edgeA.x, fs_in.edgeA.y );
        d = min( d, fs_in.edgeA.z );
    }
    else
    {
        // Handle configuration where screen space projection breaks down
        // Compute and compare the squared distances
        vec2 AF = gl_FragCoord.xy - fs_in.edgeA.xy;
        float sqAF = dot( AF, AF );
        float AFcosA = dot( AF, fs_in.edgeA.zw );
        d = abs( sqAF - AFcosA * AFcosA );

        vec2 BF = gl_FragCoord.xy - fs_in.edgeB.xy;
        float sqBF = dot( BF, BF );
        float BFcosB = dot( BF, fs_in.edgeB.zw );
        d = min( d, abs( sqBF - BFcosB * BFcosB ) );

        // Only need to care about the 3rd edge for some configurations.
        if ( fs_in.configuration == 1 || fs_in.configuration == 2 || fs_in.configuration == 4 )
        {
            float AFcosA0 = dot( AF, normalize( fs_in.edgeB.xy - fs_in.edgeA.xy ) );
            d = min( d, abs( sqAF - AFcosA0 * AFcosA0 ) );
        }

        d = sqrt( d );
    }

    // Blend between line color and phong color
    float mixVal;
    if ( d < lineWidth - 1.0 )
    {
        mixVal = 1.0;
    }
    else if ( d > lineWidth + 1.0 )
    {
        mixVal = 0.0;
    }
    else
    {
        float x = d - ( lineWidth - 1.0 );
        mixVal = exp2( -2.0 * ( x * x ) );
    }

    return mix( color, lineColor, mixVal );
}

void main()
{
    if (showDiff == 1.0)
        fragColor = vec4(fs_in.color, 1.0);
    else
    {
        // Calculate the color from the phong model
        vec4 color = vec4( adsModel( fs_in.position, normalize( fs_in.normal ) ), 1.0 );

        if (showLine == 1.0)
            fragColor = shadeLine(color);
        else 
            fragColor = color;
    }
}
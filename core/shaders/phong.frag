#version 150 core

// TODO: Replace with a uniform block
uniform vec4 lightPosition;
uniform vec3 lightIntensity;

// TODO: Replace with a struct
uniform vec3 ka;            // Ambient reflectivity
uniform vec3 kd;            // Diffuse reflectivity
uniform vec3 ks;            // Specular reflectivity
uniform float shininess;    // Specular shininess factor
uniform float showDiff;

in vec3 position;
in vec3 normal;
in vec3 color;

out vec4 fragColor;

vec3 adsModel( const in vec3 pos, const in vec3 n )
{
    // Calculate the vector from the light to the fragment
    vec3 s = normalize( vec3( lightPosition ) - pos );

    // Calculate the vector from the fragment to the eye position
    // (origin since this is in "eye" or "camera" space)
    vec3 v = normalize( -pos );

    // Reflect the light beam using the normal at this fragment
    vec3 r = reflect( -s, n );

    // Calculate the diffuse component
    float diffuse = max( dot( s, n ), 0.0 );

    // Calculate the specular component
    float specular = 0.0;
    if ( dot( s, n ) > 0.0 )
        specular = pow( max( dot( r, v ), 0.0 ), shininess );

    // Combine the ambient, diffuse and specular contributions
    return lightIntensity * ( ka + kd * diffuse + ks * specular );
}

void main()
{
    if (showDiff == 1.0)
        fragColor = vec4(color, 1.0);
    else    
        fragColor = vec4( adsModel( position, normalize( normal ) ), 1.0 );
}
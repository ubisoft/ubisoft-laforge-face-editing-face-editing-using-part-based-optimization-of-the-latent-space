#version 330 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec3 vertexColor;
in vec2 vertexTexCoord;

out EyeSpaceVertex {
    vec3 position;
    vec3 normal;
    vec3 color;
    vec2 uv;
} vs_out;

uniform mat4 modelView;
uniform mat3 modelViewNormal;
uniform mat4 mvp;

void main()
{
    vs_out.normal = normalize( modelViewNormal * vertexNormal );
    vs_out.position = vec3( modelView * vec4( vertexPosition, 1.0 ) );
    vs_out.color = vertexColor;
    vs_out.uv = vertexTexCoord;

    gl_Position = mvp * vec4( vertexPosition, 1.0 );
}

#version 450

layout(triangles) in;
layout (line_strip, max_vertices = 6) out;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( set = 1, binding = 0 ) uniform sampler2D normalMap;

layout( location = 0 ) in vec2 v2f_tc[];
layout( location = 1 ) in vec4 pos_world[];
layout( location = 2 ) in vec3 normal_world[];
layout( location = 3 ) in vec4 v2f_tang[];

layout( location = 0 ) out vec4 line_color;

void main() {
    // vec4 colors[3] = { vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0) };

    vec2 center_tc = (v2f_tc[0] + v2f_tc[1] + v2f_tc[2]) / 3.0;
    vec3 center_normal = (normal_world[0] + normal_world[1] + normal_world[2]) / 3.0;
    vec4 center_pos = (pos_world[0] + pos_world[1] + pos_world[2]) / 3.0;
    vec4 center_tang = (v2f_tang[0] + v2f_tang[1] + v2f_tang[2]) / 3.0;

    vec3 N = normalize(center_normal);
    vec3 T = normalize(center_tang.xyz);
    //T = normalize(T - dot(T, N) * N);
    vec3 B = normalize(cross(T, N));

    mat3 TBN = mat3(T, B, N);

    // transform N
    N = TBN * (texture(normalMap, center_tc).rgb * 2.0 - 1.0);

    // gl_Position = uScene.P * uScene.V * center_pos;
    // line_color = vec4(1.0, 0.0, 0.0, 1.0);
    // EmitVertex();
    // gl_Position = uScene.P * uScene.V * (center_pos + vec4(T * 0.1, 0.0));
    // line_color = vec4(1.0, 0.0, 0.0, 1.0);
    // EmitVertex();
    // EndPrimitive();

    // gl_Position = uScene.P * uScene.V * center_pos;
    // line_color = vec4(0.0, 1.0, 0.0, 1.0);
    // EmitVertex();
    // gl_Position = uScene.P * uScene.V * (center_pos + vec4(B * 0.1, 0.0));
    // line_color = vec4(0.0, 1.0, 0.0, 1.0);
    // EmitVertex();
    // EndPrimitive();

    gl_Position = uScene.P * uScene.V * center_pos;
    line_color = vec4(0.0, 0.0, 1.0, 1.0);
    EmitVertex();
    gl_Position = uScene.P * uScene.V * (center_pos + vec4(N * 0.1, 0.0));
    line_color = vec4(0.0, 0.0, 1.0, 1.0);
    EmitVertex();
    EndPrimitive();

}
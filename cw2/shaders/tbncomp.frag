#version 450

layout( location = 0 ) in vec2 v2f_tc;
layout( location = 1 ) in vec4 pos_world;
layout( location = 2 ) in mat3 tbn;

layout( location = 0) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( set = 1, binding = 0 ) uniform Light
{
    vec4 pos;
    vec4 color;
} uLight;

layout( set = 2, binding = 0 ) uniform sampler2D baseColor;
layout( set = 2, binding = 1 ) uniform sampler2D metalness;
layout( set = 2, binding = 2 ) uniform sampler2D roughness;
layout( set = 2, binding = 3 ) uniform sampler2D alphaMask;
layout( set = 2, binding = 4 ) uniform sampler2D normalMap;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}  

float NormalDF(float NdotH, float roughness)
{
    float ap = 2.0 / (roughness * roughness * roughness * roughness + 0.001) - 2.0;
    return (ap + 2.0) / (2.0 * PI) * pow(NdotH, ap);
}

float GeometryDF(float NDotH, float NDotL, float VDotH, float NDotV, float roughness)
{
    float left = 2.0 * NDotH * NDotV / VDotH;
    float right = 2.0 * NDotH * NDotL / VDotH;
    return min(1.0, min(left, right));
}

void main()
{   
    if (texture(alphaMask, v2f_tc).a < 0.5)
    {
        discard;
    }
    vec3 albedo = texture(baseColor, v2f_tc).rgb;
    float metallic = texture(metalness, v2f_tc).r;
    float roughness = texture(roughness, v2f_tc).r;

    // NORMAL MAP
    vec3 N = normalize(tbn * (texture(normalMap, v2f_tc).rgb * 2.0 - 1.0));
    if (textureSize(normalMap, 0).x == 1.0) {
        N = normalize(tbn[2]);
    }

    // calculate view and light direction
    vec3 L = normalize(vec3(uLight.pos - pos_world));
    vec3 V = normalize(vec3(uScene.camPos - pos_world));

    vec3 H = normalize(V + L);

    float NDotH = max(dot(N, H), 0.0);
    float NDotL = max(dot(N, L), 0.0);
    float VDotH = max(dot(V, H), 0.0);
    float NDotV = max(dot(N, V), 0.0);

    // brdf
    float NDF = NormalDF(NDotH, roughness);        
    float G   = GeometryDF(NDotH, NDotL, VDotH, NDotV, roughness);  
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);    
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);       

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 specular     = (NDF * G * F) / (4.0 * NDotV * NDotL + 0.001);

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);                
    vec3 radiance = uLight.color.rgb;
    vec3 color = (kD * albedo / PI + specular) * NdotL * radiance + albedo * 0.02;

    oColor = vec4(color, 1.0);
}

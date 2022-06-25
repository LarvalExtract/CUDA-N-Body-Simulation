Texture2D ParticleTexture : register(t0);
SamplerState ParticleTextureSampler : register(s0);

cbuffer CudaParams
{
	float4 ParticleParams;
}

struct VS_OUT
{
	float4 Position : SV_Position;
	float4 Colour : COLOR0;
};

VS_OUT VSMain(uint vertexId : SV_VertexID)
{
	const float2 uv = float2(vertexId % uint(ParticleParams.x), vertexId / uint(ParticleParams.y));
	const float4 particle = ParticleTexture.SampleLevel(ParticleTextureSampler, uv / ParticleParams.xy, 0);

	const float2 particlePos = particle.xy;
	float speed = particle.z / 350.0;
	float mass = particle.w;
	
	VS_OUT output;
	output.Position = float4(particlePos.x / ParticleParams.z, particlePos.y, 0.0f, 1.0f);
	output.Colour = float4(
		smoothstep(0.35f, 1.0f, speed), 
		smoothstep(0.28f, 1.0f, speed), 
		smoothstep(0.25f, 1.0f, speed), 
		1.0f) * mass;
	return output;
}

float4 PSMain(in VS_OUT input) : SV_Target
{
	return input.Colour;
}
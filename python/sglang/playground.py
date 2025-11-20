from sglang.multimodal_gen import DiffGenerator


model_path_qwen = "Qwen/Qwen-Image"
model_path_nemotron = "/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/checkpoints/afrimi_ds_r1_ckpt/models/Nemotron-Diffusion-8B"

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path=model_path_nemotron,
        num_gpus=1, 
        trust_remote_code=True # Adjust based on your hardware
    )

    # Provide a prompt for your video
    prompt = "a boss  gives his employees candy."

    # Generate the video
    image = generator.generate(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_images/",  # Controls where videos are saved
        save_output=True,
        height=512,
        width=512
    )

if __name__ == '__main__':
    main()
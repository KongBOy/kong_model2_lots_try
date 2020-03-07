import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from step1_data_pipline import get_dataset
from step2_kong_model import Generator, Discriminator


def generator_loss(disc_generated_output, gen_output, target):
    LAMBDA = 100

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(test_db, model, test_input, tar, epoch=0, result_dir="."):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    plt.savefig(result_dir + "/" + "epoch_%02i-result.png"%epoch)

# for example_input, example_target in test_db.take(1):
#     generate_images(generator, example_input, example_target)

#######################################################################################################################################
@tf.function()
def train_step(generator, discriminator,generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, \
        gen_gan_loss  , \
        gen_l1_loss  = generator_loss(disc_generated_output, gen_output, target)
        disc_loss    = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients     = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


# def fit(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, checkpoint, train_db, epochs, test_db):
#     for epoch in range(epochs):
#         print("Epoch: ", epoch)
#         start = time.time()
#         for example_input, example_target in test_db.take(1):
#             generate_images(test_db, generator, example_input, example_target, epoch)
        
#         # Train
#         for n, (input_image, target) in train_db.enumerate():
#             print('.', end='')
#             if (n+1) % 100 == 0:
#                 print()
#             train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
#         print()

#         # saving (checkpoint) the model every 20 epochs
#         if (epoch + 1) % 20 == 0:
#             checkpoint.save(file_prefix = checkpoint_prefix)

#         print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
#                                                             time.time()-start))
    # checkpoint.save(file_prefix = checkpoint_prefix)


#######################################################################################################################################
if(__name__=="__main__"):
    from step1_data_pipline import get_dataset
    from build_dataset_combine import Check_dir_exist_and_build
    import os
    import time

    DATA_AMOUNT = 400
    BATCH_SIZE = 1
    
    db_dir  = "datasets"
    db_name = "facades"

    start_time = time.time()
    train_db, test_db = get_dataset(db_dir=db_dir, db_name=db_name,batch_size=BATCH_SIZE, data_amount=DATA_AMOUNT)

    ##############################################################################################################################
    start_time = time.time()
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    print("build model cost time:", time.time()-start_time)
    ##############################################################################################################################
    epochs = 150
    import datetime
    log_dir="logs/"

    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # fit(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, checkpoint, train_db, EPOCHS, test_db)


    result_dir = "result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    Check_dir_exist_and_build(result_dir)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        start = time.time()
        for example_input, example_target in test_db.take(1):
            generate_images(test_db, generator, example_input, example_target, epoch, result_dir)
        
        # Train
        for n, (input_image, target) in train_db.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
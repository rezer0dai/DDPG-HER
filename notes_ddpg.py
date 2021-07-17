{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "3M4poI6KmBxU",
    "outputId": "4b48f275-14d9-496d-9e1d-d594f7bfb018"
   },
   "outputs": [],
   "source": [
    "#!git clone https://github.com/alirezakazemipour/DDPG-HER/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "DJYTd2ZKmEoY",
    "outputId": "36e57ddf-5746-48d4-e0bf-a4818a28371b"
   },
   "outputs": [],
   "source": [
    "#!git clone https://github.com/qgallouedec/panda-gym\n",
    "\n",
    "#!pip install -e panda-gym"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "cQUJ6zL0mIN0",
    "outputId": "917a7696-950e-493b-f863-b4c28970fcf7"
   },
   "outputs": [],
   "source": [
    "#!pip install tensorboard_logger"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "6KoztcqVmLox",
    "outputId": "f4595dfc-0662-4871-8988-403e6db64457"
   },
   "outputs": [],
   "source": [
    "#!pip install matplotlib psutil mpi4py gym"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "u4rO1WzEmYc3"
   },
   "outputs": [],
   "source": [
    "#import sys\n",
    "\n",
    "#libs = [\"DDPG-HER\", \"panda-gym\"]\n",
    "#for lib in libs:\n",
    "#    sys.path.append(lib)\n",
    "#    sys.path.append(\"/content/\"+lib)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "cXw-z2u3mjhJ"
   },
   "outputs": [],
   "source": [
    "import main"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "DFdnbp8dqNCe"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.8/dist-packages/gym-0.15.7-py3.8.egg/gym/logger.py:30: UserWarning: \u001b[33mWARN: Box bound precision lowered by casting to float32\u001b[0m\n",
      "  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))\n"
     ]
    }
   ],
   "source": [
    "import open_gym\n",
    "env = open_gym.make_env(render=True, colab=False, env_name=\"panda-pusher\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "k0D1dZ6BqwKd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch:0| Running_reward:-50.000| EP_reward:-50.000| Memory_length:100| Duration:377.961| Actor_Loss:0.717| Critic_Loss:0.005| Success rate:0.000| 3.7/7.7 GB RAM\n",
      "Epoch:1| Running_reward:-50.000| EP_reward:-50.000| Memory_length:200| Duration:561.105| Actor_Loss:1.147| Critic_Loss:0.007| Success rate:0.000| 3.7/7.7 GB RAM\n",
      "Epoch:2| Running_reward:-50.000| EP_reward:-50.000| Memory_length:300| Duration:768.853| Actor_Loss:1.425| Critic_Loss:0.010| Success rate:0.000| 3.8/7.7 GB RAM\n",
      "Epoch:3| Running_reward:-50.000| EP_reward:-50.000| Memory_length:400| Duration:792.444| Actor_Loss:2.200| Critic_Loss:0.016| Success rate:0.000| 3.8/7.7 GB RAM\n",
      "Epoch:4| Running_reward:-50.000| EP_reward:-50.000| Memory_length:500| Duration:1167.901| Actor_Loss:2.586| Critic_Loss:0.056| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:5| Running_reward:-50.000| EP_reward:-50.000| Memory_length:600| Duration:1343.244| Actor_Loss:2.491| Critic_Loss:0.019| Success rate:0.000| 3.9/7.7 GB RAM\n",
      "Epoch:6| Running_reward:-50.000| EP_reward:-50.000| Memory_length:700| Duration:823.737| Actor_Loss:2.718| Critic_Loss:0.081| Success rate:0.000| 3.9/7.7 GB RAM\n",
      "Epoch:7| Running_reward:-49.917| EP_reward:-50.000| Memory_length:800| Duration:266.646| Actor_Loss:3.450| Critic_Loss:0.058| Success rate:0.000| 3.9/7.7 GB RAM\n",
      "Epoch:8| Running_reward:-50.000| EP_reward:-50.000| Memory_length:900| Duration:268.791| Actor_Loss:4.013| Critic_Loss:0.153| Success rate:0.000| 3.9/7.7 GB RAM\n",
      "Epoch:9| Running_reward:-49.799| EP_reward:-50.000| Memory_length:1000| Duration:267.368| Actor_Loss:4.319| Critic_Loss:0.123| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:10| Running_reward:-50.000| EP_reward:-50.000| Memory_length:1100| Duration:266.390| Actor_Loss:5.457| Critic_Loss:0.518| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:11| Running_reward:-49.971| EP_reward:-50.000| Memory_length:1200| Duration:266.772| Actor_Loss:5.542| Critic_Loss:0.392| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:12| Running_reward:-16.963| EP_reward:-49.000| Memory_length:1300| Duration:266.374| Actor_Loss:5.294| Critic_Loss:0.383| Success rate:0.100| 4.0/7.7 GB RAM\n",
      "Epoch:13| Running_reward:-49.604| EP_reward:-50.000| Memory_length:1400| Duration:266.306| Actor_Loss:4.797| Critic_Loss:1.529| Success rate:0.100| 4.0/7.7 GB RAM\n",
      "Epoch:14| Running_reward:-50.000| EP_reward:-50.000| Memory_length:1500| Duration:268.101| Actor_Loss:6.037| Critic_Loss:0.396| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:15| Running_reward:-50.000| EP_reward:-50.000| Memory_length:1600| Duration:267.865| Actor_Loss:6.012| Critic_Loss:0.302| Success rate:0.000| 4.0/7.7 GB RAM\n",
      "Epoch:16| Running_reward:-50.000| EP_reward:-50.000| Memory_length:1700| Duration:266.829| Actor_Loss:5.820| Critic_Loss:1.013| Success rate:0.000| 4.1/7.7 GB RAM\n",
      "Epoch:17| Running_reward:-17.113| EP_reward:-50.000| Memory_length:1800| Duration:270.819| Actor_Loss:6.116| Critic_Loss:0.701| Success rate:0.100| 4.1/7.7 GB RAM\n",
      "Epoch:18| Running_reward:-47.259| EP_reward:-50.000| Memory_length:1900| Duration:270.326| Actor_Loss:6.072| Critic_Loss:0.721| Success rate:0.000| 4.1/7.7 GB RAM\n",
      "Epoch:19| Running_reward:-49.912| EP_reward:-50.000| Memory_length:2000| Duration:273.629| Actor_Loss:6.836| Critic_Loss:0.754| Success rate:0.000| 4.1/7.7 GB RAM\n",
      "Epoch:20| Running_reward:-15.677| EP_reward:-50.000| Memory_length:2100| Duration:271.631| Actor_Loss:5.650| Critic_Loss:1.902| Success rate:0.300| 4.1/7.7 GB RAM\n",
      "Epoch:21| Running_reward:-27.682| EP_reward:-11.000| Memory_length:2200| Duration:275.800| Actor_Loss:5.947| Critic_Loss:0.778| Success rate:0.500| 4.1/7.7 GB RAM\n",
      "Epoch:22| Running_reward:-48.872| EP_reward:-19.000| Memory_length:2300| Duration:276.219| Actor_Loss:6.551| Critic_Loss:0.530| Success rate:0.400| 4.1/7.7 GB RAM\n",
      "Epoch:23| Running_reward:-48.995| EP_reward:-50.000| Memory_length:2400| Duration:280.309| Actor_Loss:5.940| Critic_Loss:0.857| Success rate:0.300| 4.2/7.7 GB RAM\n",
      "Epoch:24| Running_reward:-30.285| EP_reward:-6.000| Memory_length:2500| Duration:284.504| Actor_Loss:5.604| Critic_Loss:0.632| Success rate:0.500| 4.2/7.7 GB RAM\n",
      "Epoch:25| Running_reward:-49.131| EP_reward:-50.000| Memory_length:2600| Duration:282.896| Actor_Loss:5.940| Critic_Loss:0.372| Success rate:0.300| 4.2/7.7 GB RAM\n",
      "Epoch:26| Running_reward:-49.511| EP_reward:-50.000| Memory_length:2700| Duration:286.414| Actor_Loss:6.597| Critic_Loss:0.511| Success rate:0.200| 4.2/7.7 GB RAM\n",
      "Epoch:27| Running_reward:-48.122| EP_reward:-8.000| Memory_length:2800| Duration:475.415| Actor_Loss:5.237| Critic_Loss:0.628| Success rate:0.600| 4.3/7.7 GB RAM\n",
      "Epoch:28| Running_reward:-44.042| EP_reward:-50.000| Memory_length:2900| Duration:388.196| Actor_Loss:5.807| Critic_Loss:0.805| Success rate:0.500| 4.0/7.7 GB RAM\n"
     ]
    }
   ],
   "source": [
    "agent = main.train(env, env)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "HoYROjMyq2vU"
   },
   "outputs": [],
   "source": [
    "!cp DDPG-HER drive/MyDrive/ -r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Euh9exoHrIYH"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Untitled0.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

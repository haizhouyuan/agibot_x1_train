明白了，我将分析你上传的视频中机器人行走的表现问题，并结合你 GitHub 项目（haizhouyuan/agibot\_x1\_train）中的 PPO 强化学习训练代码，对奖励函数、环境设计和策略网络进行评估，找出影响稳定性的因素，并提出改进建议。完成后会将分析结果呈现给你。


## 视频行走问题观察与分析

视频中可以明显看到机器人步态存在 **稳定性不足** 的现象，包括以下几个方面：

* **左右晃动（侧向不稳）**：机器人行走时躯干左右摇摆较大，说明其对横向平衡的控制不足。每迈出一步时身体在侧向摆动，这可能是由于策略没有充分约束重心在支撑脚投影内所致，也可能是由于控制频率或延迟导致调整不及时。

* **步态不对称**：从视频看出左右腿步幅或步态周期不一致，表现为一侧步子大、一侧步子小，或者左右支撑时间不同。这种 **不对称** 可能会引起额外的左右晃动和不稳定。不对称步态通常反映奖励函数未能鼓励左右动作对称，或策略在训练中陷入局部最优步态模式。

* **前倾现象**：机器人行走时身体有明显的 **前倾**，似乎重心总是偏向前方，需要不断加快步伐“追赶”重心。这表明策略可能没有充分惩罚身体俯仰角偏离竖直，或奖励函数中过于鼓励向前速度而忽视了上身姿态的稳定。前倾也可能由于停止和启动时缺乏平衡控制所致。

* **脚步失衡**：步态中可见 **脚步落地不稳**，如有时脚尖先着地、脚滑动或双脚同时离地。视频中机器人偶尔出现步态紊乱的迹象，说明现有策略对着地冲击、脚底滑移等情况缺乏充分抑制。比如，脚落地时摩擦力不足导致打滑，或策略未很好地控制每一步的节奏，出现临空或过长支撑等问题。

上述问题综合来看，机器人虽然能够行走向前，但步态晃动较大、不够协调自然，存在潜在跌倒风险。下面结合训练代码的设计，分析这些现象可能的根源。

## 强化学习训练设置检查

为找出步态不稳的原因，我们审查了 GitHub 项目 **AgiBot X1** 强化学习训练的关键设置，包括仿真环境、奖励函数、观测/动作空间和策略网络架构。

### 环境建模与控制参数

训练在 Isaac Gym 仿真环境中进行，使用PhysX物理引擎。机器人采用 **PD控制** 模式（control\_type='P'）——策略输出关节目标角度，由PD控制器跟踪执行。关节刚度和阻尼在配置中定义，如髋、膝、踝各关节的 \$K\_p\$ 和 \$K\_d\$ 值。例如膝关节刚度设为100，阻尼10，保证腿部有一定支撑力但也不会过于生硬。控制频率为 **100Hz**（物理引擎仿真步长dt=0.001s，每10个仿真步更新一次控制，即decimation=10）。100Hz的控制频率与实际机器人控制频率类似，可实时响应姿态变化。

仿真中地形主要为**平地**(mesh\_type='trimesh')，摩擦系数设为0.6。这些参数与真实环境接近，但如果真实机器人地面的摩擦或不平程度有差异，可能影响步态稳定。环境中还允许**随机推挤**机器人以训练其抗扰（每隔4秒给予随机冲量）。此外，配置中**域随机化**参数非常丰富，包括摩擦系数、质心位置、关节摩擦/阻尼、执行延迟等随机变化。这些设置（在 **v1.5 域随机化** 分支中引入）旨在提高策略的泛化和鲁棒性。然而，需要确保训练中真的启用了这些随机化（例如配置`randomize_friction=True`等均生效），否则现实与仿真的差异可能导致实际步态不稳。

### 奖励函数设计

该项目的奖励函数设计相当复杂细致，目的是鼓励稳定直立、规律行走的行为。奖励函数包含多项指标，主要包括：

* **运动规律与步态周期**：使用 **步态周期性奖励**，鼓励机器人以固定周期交替迈步。代码中设置了目标步态周期为0.7秒。例如，引入 **脚离地时间奖励**（feet\_air\_time）鼓励每条腿有足够的摆空时间；**足接触数奖励**根据步态相位判断当前应有哪只脚接地，若符合预期则奖励1分，反之惩罚0.3分（例如单支撑期应只有一脚落地，两脚都落地则视为不符节奏）。这些机制保证机器人的步态节奏接近人形机器人的自然走路模式。

* **对称性与协调**：v1.3版本增加了**对称性奖励**。虽然代码未直接给出“对称性奖励”项名字，但通过设置左右腿**目标摆动终点**对称（`final_swing_joint_delta_pos`为左右各6个关节目标偏移）并引入 **参考关节位置奖励**（ref\_joint\_pos），鼓励左右腿在步态周期内达到这些对称的目标姿态。简言之，就是要求机器人每一步摆动腿达到预定前伸高度/幅度、支撑腿适当弯曲，从而左右步态镜像。对称性奖励的引入正是为了解决我们在视频中看到的步态不对称问题。

* **稳定性与平衡**: 为了直立稳定，奖励函数包含 **姿态平衡** 项。例如**躯干姿态奖励**(orientation)根据机器人躯干的横滚和俯仰角偏差来奖励平直姿态；**基座高度奖励**(base\_height)确保机器人髋部高度接近期望值0.61m，避免躯干过高或过低导致不稳；**角动量惩罚**(未直接在此处列出，但v1.2版本说明添加了角动量惩罚)用于约束全身晃动。**脚滑动惩罚**(foot\_slip)根据脚接触时足部速度和接触力来惩罚打滑。例如当脚落地接触力>阈值且水平速度不为零则视为滑动，从而产生负奖。**接触力惩罚**(feet\_contact\_forces)则限制脚踝承受的冲击力大小；超过700N的部分会产生惩罚，使策略学会缓冲着地。还有**关节默认位置奖励**(default\_joint\_pos)鼓励关节姿态贴近默认站姿，特别减少髋关节yaw/roll方向偏差。**关节速度/加速度惩罚**和**能耗惩罚**（dof\_vel, dof\_acc, torques项）则抑制过于剧烈的动作和能源浪费。此外，当机器人静止（速度指令接近0）时，有**静止奖励**(stand\_still=2.5)，鼓励它稳稳站立不东倒西歪。

* **指令跟踪**：为了让机器人按需行走，奖励包含 **速度跟踪** 项。包括线速度和角速度跟踪误差的奖励（tracking\_lin\_vel和tracking\_ang\_vel），其中线速度跟踪权重较高(1.8)鼓励匹配期望前进/侧移速度，角速度权重1.1用于跟踪期望转向速度。如果机器人速度超过指令或者偏离，会被惩罚。此外竖直方向速度和不必要的横摆被视作不匹配速度（vel\_mismatch）给予惩罚。通过这些，使机器人既能走到目标速度又不乱晃。

综合来看，奖励函数已经考虑到了**周期、对称、平衡、能耗**等多方面因素，设计相当全面。特别是后续版本逐步加入了**周期性奖励**(v1.1)、**角动量惩罚**(v1.2)、**对称性奖励**(v1.3)并取得更自然的步态。然而，权重如何平衡以及是否还有未涵盖的因素（如横向稳定的特定指标）值得关注。

### 观测空间与动作空间

**观测空间**决定了策略能感知的状态信息。根据代码，演员网络每个环境时刻接收一个大小为47的观测向量。主要包含：

* **步态相位和速度指令**：前2维是当前步态相位的 \$\sin\$ 和 \$\cos\$ 值，接下来3维是期望机体的\$(v\_x, v\_y, \omega\_z)\$，即前进速度、横移速度和旋转（yaw角速度）指令。将步态相位显式输入，有助于策略了解在步态周期中的位置，从而配合周期性奖励形成**内隐的步态时钟**。

* **自身状态反馈**：包含关节角度和角速度，以及躯干姿态和角速度等。代码显示12个关节角偏离默认值（q）和12个关节角速度（dq）经过适当缩放后作为观测。还有上一时刻的12维动作输出（actions）反馈给网络，让策略了解之前关节命令，有助于平滑控制。此外，观测包括机体角速度（3维陀螺仪）和机体姿态欧拉角（横滚、俯仰、偏航3维）。姿态信息经过裁剪（例如只取小角度范围）并缩放后提供。**注意**：观测中并没有直接包括“双足接触状态”或“足底压力”等信息——这些接触数据仅存在于“Privileged”扩展观测中供评论家(Critic)使用。也就是说，策略本身需通过躯干运动和关节状态间接推断接地情况。

* **历史帧堆叠**：配置中 `frame_stack=66` 表明策略实际接收的是**66帧历史观测序列**（每帧47维，叠加形成长历史）。这一设计在没有RNN时提供一定的时序信息，使策略能依据一段时间的运动历史做出决策（例如判断当前是在摆动相还是支撑相）。3102维输入正是66×47形成的历史观测序列。另外还有一个短历史(short\_frame\_stack=5)和特权观测历史(c\_frame\_stack=3)用于评论家，但演员侧主要用长历史。长序列输入通过特殊的卷积网络进行处理以提取时间特征（配置中为一维卷积核kernel\_size=\[6,4]两层、通道数\[32,16]等，将66×47的输入压缩到64维表示）。

**动作空间**则是策略输出的12维关节目标值。表明机器人有12个可控关节，自定义的动作是每个关节相对于默认角度的增量。配置`action_scale=0.5`意味着输出范围\[-1,1]将映射为±0.5弧度的关节变化量，加到预设默认角度上。例如默认姿态中左髋Pitch为0.4rad，那么动作+1会让髋Pitch达到约0.9rad，从而迈腿更前。这样的动作参数化确保动作幅度适中，防止RL输出过大导致关节进入极限。控制在PD内环作用下执行，使实际关节跟踪这些目标角度。由于策略每0.01秒输出一次动作，在仿真和实际中能及时调整步态细节。

总体来说，观测空间涵盖了机器人本体状态和期望指令，但**缺失直接的足部接触/滑移反馈**给策略。这可能影响其对平衡的感知。而动作空间的设计考虑了硬件限制，采用关节角度增量形式易于PD控制实现。

### 策略网络架构

策略采用 **PPO算法**，配以特制的神经网络结构。根据配置，演员网络为多层感知机(MLP)加上历史信息处理模块：

* **主体网络**：三层全连接隐藏层，尺寸为512→256→128个单元，用于处理当前时刻观测。评论家网络规模稍大（768→256→128）以处理更多信息。隐藏层使用较小的初始权重噪声(std=1.0)并结合归一化等，保证训练稳定。

* **长历史处理**：由于观测含66帧序列，网络引入一套**卷积模块**（“long\_history CNN”）来提取时间特征。配置显示卷积核大小6和4，步幅3和2，最后输出64维的历史特征。这相当于一个小型一维CNN在时间维度上提取运动模式（类似步态周期特征），再与当前观测综合。这种架构在没有循环网络时可有效捕捉步态相位、震荡等时序信息。

* **LSTM拓展**：项目在v1.4版本中引入了 **LSTM网络** 支持。LSTM是一种循环神经网络，擅长处理长期依赖关系。如果采用LSTM版本，策略就不需要叠加66帧输入，而是通过隐藏状态记忆历史。当前代码结构中，Runner类可以选择不同的ActorCritic实现，例如切换到 `ActorCriticLSTM` 类（从提交记录看v1.4增加了LSTM支持）。使用LSTM有望让策略更连贯地应对平衡，如提前预判摆动和修正倾斜。不过在我们分析的视频对应训练版本中，可能仍在使用MLP+CNN历史的架构 (因为v1.4于7月4日才完成训练，尚不确定是否部署)。

* **其他**：PPO超参数采用了较小的学习率1e-5、熵系数0.001以保持足够探索。每轮迭代采样24步、4个mini-batch，折扣因子\$\gamma=0.994\$偏保守。这些参数配置说明训练过程较为稳定，但学习速率慢也可能导致需要更多迭代才能充分收敛。

综上，训练设置在**仿真、奖励、观测和网络**各方面做了精心设计，旨在得到稳定的行走策略。即便如此，实际观察到的问题可能源于某些设计细节与现实情况的不匹配或权衡不足，下面我们进一步分析训练设计如何影响步态表现。

## 训练设计与步态不稳定性的关联分析

结合上述训练设置，我们认为以下因素可能导致了视频中观察到的步态不稳问题：

* **奖励函数侧重点与现实平衡**：尽管奖励考虑了姿态和对称，但其权重设置可能使策略更关注前进速度而相对忽视横向稳定。例如，线速度跟踪权重大（1.8）而横滚姿态奖励权重为1.0。策略可能为了优化前进而允许一定横向晃动和前倾，只要不被过度惩罚。在仿真中也许仍能行走，但现实中横向晃动更危险。**角动量惩罚**的加入表明开发者注意到了大幅晃动的问题，但若权重不够高，策略可能仍接受左右摇摆的步态来获得步幅或速度上的奖励。换言之，目前的奖励函数可能**不足以抑制侧向晃动**，导致机器人需要左右摇摆来维持动态平衡（这是许多人形机器人常见的“八字步”平衡策略，但过度摇摆会显得不稳）。

* **对称性收敛不足**：**步态不对称**可能暗示策略没有完全利用对称性奖励，或者在训练中陷入了不对称的局部最优。可能原因包括：对称性相关奖励（如\_ref\_joint\_pos\_或\_feet\_distance\_等）的权重偏低，相对于速度奖励不够显著；或者训练初期随机性导致左右腿分工差异，未充分纠正。这也可能与**早熟收敛**有关——如果在引入对称性奖励前模型已经学会一种略不对称但稳定的走法，后来奖励调整也难以完全纠正这一习惯。尽管v1.3版本据报告达到了“自然步态”，实际部署中仍可能存在轻微不对称，需要进一步增大对称性奖励或采用新的对称约束方法。

* **观测不足导致平衡反应滞后**：策略没有直接的足底反馈，这意味着它只能通过躯干倾角和关节状态来**间接感知失衡**。现实中，人形机器人通常依赖IMU检测倾斜，以及脚传感器检测载荷变化来判断失衡并快速做出反应。而我们的策略观测虽然有IMU角速度和姿态，但缺少加速度或足接触信息。如果机器人滑脚或出现脚未完全落地，策略可能要等到躯干产生明显倾斜后才能察觉，这就**滞后**了一步，导致视频中看到摇晃幅度较大。此外，训练中大量的历史帧输入或LSTM本应帮助预测平衡，但如果关键感知缺失，策略在横向平衡上仍可能反应不够敏捷。这解释了为何机器人在每步支撑转换时左右晃：策略可能并不知道当前支撑面的支持情况，无法提前调整下一个步态去抵消晃动。

* **动作频率与执行误差**：仿真中100Hz控制相当理想，但真实机器人即使控制频率高，也存在传感和执行延迟。训练代码虽有模拟 **延迟**（如`add_lag=True`随机施加5\~40步的延迟）来逼近现实，但延迟建模是否充分仍是疑问。如果实际系统延迟比仿真更大或者不同步，策略的快速小幅调整可能在硬件上滞后，表现为节奏不稳、脚着地时机不准。这也能导致左右晃动或脚步错乱，因为反馈控制相对于实际姿态变化有相移。此外，若真实机器人执行动作时力控性能与仿真PD模型不一致（例如关节阻尼、摩擦），策略计算的步态推力可能过大或不足，导致落地不稳。虽然域随机化对这些模型不确定性有所覆盖，但真实性能的偏差仍可能让策略**没有完全适应现实**。

* **策略复杂度与训练充分性**：AgiBot X1属于高维控制（34个自由度，12个主要驱动关节参与行走），策略网络复杂且训练任务困难。虽然项目进行了上万次迭代训练并取得一定成功，但**训练是否足够充分**值得考量。学习率较低（1e-5）虽稳定但收敛慢，可能需要更多迭代来精调平衡能力。如果训练过早停止在一个次优策略上，可能留下例如小幅度晃动这样的瑕疵。另外，采用PPO等随机梯度法，本身结果有方差，不排除当前策略只是**其中一个次优解**（比如更倾向摇摆行走）。多次训练取最优或加入更丰富场景（如不同速度指令下的训练）才能减少此类不佳行为。

* **仿真与现实差异**：如果视频是机器人真实环境测试，那么**Sim2Real误差**往往是首要原因。即使加入了域随机化，仍可能有未覆盖的差异。例如摩擦系数0.6只是模拟值，地面的实际摩擦和不平均可能不同；机器人的质量分布或关节死区等微小差别，都可能让原本在仿真中边缘稳定的步态在现实中放大为明显晃动。尤其横向平衡和前后倾，是人形机器人最敏感的两个方向，任何模型误差都可能导致策略输出无法完全抵消重心偏移。因此，如果域随机化范围不够大或未涵盖某些参数（比如没有模拟传感器噪声形态、没有在不同地面测试），现实中就会出现步态发飘、落地打滑等问题。**域随机化**在v1.5版本被强调，表明开发者也认识到需要更强的鲁棒性来应对现实环境。

总的来说，训练设计本身比较全面，但**步态平衡是极其复杂的任务**。一些细节（观测、奖励权重、延迟模拟等）的不足，叠加现实的不可预知因素，最终导致了视频中看到的那些不稳表现。所幸，这些问题多数可以通过改进训练来缓解，下面我们针对性地提出建议。

## 改进训练的具体建议

针对以上分析的原因，我们建议从以下几方面改进训练，以提升机器人行走的稳定性和自然度：

* **优化奖励函数权重与项设计**：适当调整奖励函数中与平衡相关的权重，使策略更重视稳定行走而非纯粹追求速度。具体来说，可以**提高姿态稳定项**（如orientation、base\_height、base\_acc）的权重，加大对横滚/俯仰偏差的惩罚力度，促使机器人减少左右摇晃和前倾幅度。同时，**增加新的平衡指标**：例如引入**质心投影误差惩罚**（确保质心落在支撑多边形内），或直接加入**横向角动量惩罚**（约束左右摇摆的惯性）等。脚滑动惩罚和接触力惩罚可以再加强（目前足滑惩罚权重仅-0.1，可适当加大），让策略学会放缓脚落地速度、避免打滑。对于**步态对称性**，若仍有不良倾向，可显式增加一项奖励：比如计算左右腿步幅或支撑时长的差异，差异越小奖励越高，从而数值上直接鼓励对称。总之，重新平衡奖励函数，使其在\*\*“稳”**上给予更高回报，在**“快”\*\*上有所克制，训练出的策略将倾向更稳健的步态。

* **丰富观测信息**：考虑在策略观测空间中加入关键的**平衡感知信号**。首先，建议加入 **足部接触状态**（例如左右脚是否接地的二元值或接触力大小）。实际上，评论家已经有这部分信息，完全可以将其也提供给演员网络。这将显著帮助策略判断当前支撑腿，及时在单脚支撑时做出姿态调整，避免因为不确定是否落地而过度倾斜。其次，可以加入**IMU加速度**信息，尤其是躯干的线加速度或倾角变化率，让策略更早地感知到失衡趋势。人类在跌倒前会感觉到加速度异常而迅速反应，机器人也需要类似的快速前馈信号。再次，可以输入**足底相对高度**（若有传感器）或使用腿部关节状态推算的足端高度，以判断脚何时离地/触地。虽然高度测量在平地可由几何计算获得，但代码中`measure_heights`目前关闭，可以考虑开启并将适当的高度差值输入观测。总的来说，**增加传感维度**有助于策略更准确地掌握动态，提高对失衡的敏感度。不过也要注意控制观测维度增长，必要时可以通过自编码器或注意力机制提取这些新增信息的要点，避免高维观测影响学习。

* **策略网络改进**：利用已有的 **LSTM网络**架构(v1.4)来替代单纯堆叠历史帧的方式，将会显著提升策略对时序的理解和记忆。LSTM能依据长期隐藏状态来平滑输出，对付步态周期内的微小扰动更有效。如尚未在行走策略中使用LSTM，强烈建议在新一轮训练中启用，并适当增大LSTM层的规模或层数（例如使用两层LSTM）来增强记忆能力。另外，可以尝试**更深或更宽的网络**以提高策略表达能力，例如增加Actor网络隐藏单元数量。毕竟行走是复杂任务，一个128神经元的最后层可能限制了策略对非线性平衡策略的刻画。**状态估计网络**(state\_estimator)的卷积提取出的64维特征也可以考虑升级为更复杂的时序模型（如TCN或Transformer）来捕捉步态周期特征。简言之，让网络有**更强的建模时间和非线性关系的能力**，才能学到更精妙的平衡控制策略。需要注意训练更复杂网络时防止过拟合，可配合加强随机化。

* **加强域随机化与鲁棒训练**：现实与仿真的差距需要更大程度的随机化来弥补。建议在训练中进一步**扩大域随机化范围**，例如摩擦系数范围可以更广（目前0.2\~1.3或许还不够涵盖所有实际地面情况），甚至动态改变地形坡度、表面粗糙度等让策略学会适应不同条件。此外，可加入**传感噪声**和**延迟**的多样化模拟，例如随机丢弃或扰动观测数据，随机延迟控制信号等。目前配置已包含关节滞后和IMU滞后的随机抖动设置，这些应确保在训练时打开（如`add_imu_lag=True`等）。如果之前训练为了加快收敛可能降低了随机化强度，现在可以在策略已有基础上再进行**鲁棒精调**：逐步增加噪声水平到更高（如噪声level从1.5提高到2或3），并引入 **随机推挤** 训练机器人在受到更频繁、更大的侧向扰动下依然行走。这会逼迫策略学会更稳健的步态，如更低的重心、更小的步长来应对不确定性。必要时，还可以采用 **教师学生策略**（Asymmetric AC 已经有了）进一步提升鲁棒性，让教师掌握全部状态给予更精准指导。简而言之，**让训练“难度”超过未来机器人实际遇到的难度**，才能确保真正部署时步态平稳。

* **训练流程与策略调整**：在训练方法上，可以尝试 **分阶段训练（课程学习）**。例如先训练站稳和慢走，再逐渐增加目标速度和扰动。当前环境episode长达24秒可以覆盖很多步，但也可能让初期随机策略频繁跌倒难以学到东西。通过课程学习，先让机器人在静止或慢速下稳定行走几步（例如设定低速奖励、更短周期），然后逐渐提高难度，有助于策略先掌握平衡再学习快走。此外，**多策略融合**也是思路之一：可以训练多个模型（不同种子或不同奖励权重设置），选取其中**稳定性最佳**的模型部署，或者用**遗传算法**等在已有模型附近寻找更优解。最后，部署前可考虑在人形机器人常用的**平衡控制器**上做层叠，例如在RL输出基础上加一个小型PID校正躯干角度。这虽不是训练改进，但可在测试阶段作为保险，防止因意外扰动导致的失衡。

* **利用示范和模板**（高级可选）：如果有理想的人形步态数据（来自人类或优化器产生的步态），可以考虑加入**模仿学习**成分。例如在奖励中增加模仿参考轨迹的项，或预训练一个参考模型。这可以为RL提供一个良好初始策略，避免其走歪路。同时模仿自然的步态或许能直接消除不对称和多余晃动。当然，这需要有可靠的参考运动；若无，可采用简单 **参考摆动**（代码中已经给出了理想摆动关节姿态）来辅助。这属于额外增强手段，需谨慎权衡，但在步态质量要求极高时可能值得尝试。

综上所述，通过**调整奖励**突出稳定、**丰富观测**弥补感觉、**升级网络**强化记忆、**加强随机化**提高鲁棒，以及**改进训练策略**，有望显著改善AgiBot X1机器人的行走稳定性。特别地，项目已经在持续迭代版本（v1.3对称性、v1.4 LSTM、v1.5随机化），这些改进方向正对应了上述建议。当这些优化充分结合并训练收敛后，我们期待机器人能够实现 **左右对称、重心稳定** 的自然步态，减少视频中观察到的晃动前倾等问题，获得更加可靠的行走能力。

**参考资料：** 项目源码，项目文档，及相关训练配置与代码注释。

明白了，我将围绕如何在 haizhouyuan/agibot\_x1\_train 项目基础上增强机器人对多种扰动（外力、地形、控制误差）的鲁棒性展开分析，并重点优化机器人在被扰动后最短时间内恢复平衡的能力。

我将结合你的项目代码结构，具体提出训练配置、奖励函数、环境扰动机制等方面的修改建议，包括清晰的代码段调整和推荐参数设置。完成后会将结果呈现给你。


# Comprehensive Disturbance Training Design

## External Force Disturbances (Random Pushes)

To **simulate random directional pushes**, configure periodic impulse forces in the environment. In the `domain_rand` settings, enable `push_robots` and specify a shorter interval and force range. For example, set pushes about every 4 seconds with randomized directions:

* **Push Frequency:** `push_interval_s: 4` (applies a push roughly every 4s, so \~6 pushes in a 24s episode). This increases disturbance frequency compared to the default 15s.
* **Push Magnitude:** Use a moderate initial magnitude to avoid immediate failures. For instance, `max_push_vel_xy: 0.2` (m/s) for lateral pushes and `max_push_ang_vel: 0.2` (rad/s) for rotational perturbations. This sets an impulse that nudges the robot without overwhelming it.
* **Random Direction:** The `_push_robots()` function already randomizes push direction (it assigns a random XY velocity to the base). No change needed here – the push will come from any direction on the horizontal plane.
* **Non-Synchronized Pushes (optional):** To avoid all environments being pushed simultaneously, you can introduce randomness in timing. For example, apply pushes to a random subset of environments or jitter the interval. *Implementation:* within `_push_robots()`, select random `env_ids` instead of all, and call `set_actor_root_state_tensor_indexed` for those. This ensures not every robot is hit at the same simulation step, increasing training diversity.
* **Gradual Intensity:** Use curriculum to increase push strength over time. For example, start with very mild or no push and then ramp up. In code, you can define a schedule like `push_duration` or magnitude increments. The config snippet below shows an example where the first stage has `push_duration = 0` (no push), and later stages increase the push duration from 0.05 up to 0.25 seconds. The parameter `update_step = 2000*24` can be used to move to the next push intensity after a certain number of steps (e.g. after 2000 episodes):

  ```yaml
  push_robots: True  
  push_interval_s: 4           # push every 4 seconds  
  push_duration: [0, 0.05, 0.1, 0.15, 0.2, 0.25]  # gradual push duration (s)  
  update_step: 48000           # after 48000 steps, move to next duration  
  max_push_vel_xy: 0.2         # initial push velocity magnitude (m/s)  
  max_push_ang_vel: 0.2        # initial rotational push (rad/s)  
  ```

  *This design starts with no push, then introduces brief impulses, increasing to 0.25s pushes as training progresses.* By gradually intensifying pushes, the policy learns to handle small disturbances before facing larger ones.

## Terrain Disturbances (Uneven Ground)

Configure a mixture of challenging terrains so the robot encounters slopes, steps, and gaps during training:

* **Terrain Types:** Use the `terrain_dict` proportions to include various disturbances. For example, enable *stairs* and *wave* patterns that were previously zero. You might set: `"stairs up": 0.1`, `"stairs down": 0.1`, and increase `"discrete": 0.2` for random pits/blocks, while still keeping some flat ground. This exposes the robot to **slopes, pits, and raised steps** as requested. Ensure `mesh_type: 'trimesh'` is used (already set) to allow these complex terrains.
* **Terrain Parameters:** Adjust the ranges for these features. For instance:

  * `slope_range = [0, 0.2]` (increase max slope to \~11°),
  * `stair_height_range = [0.05, 0.15]` (5–15cm steps),
  * `discrete_height_range = [0.0, 0.1]` (up to 10cm obstacles or holes).
  * You can also simulate low-friction patches or different ground materials via friction randomization (already enabled with `randomize_friction: True` and a range).
* **Curriculum on Terrain:** Set `terrain.curriculum: True` to gradually increase difficulty if desired. With curriculum on, the environment can start on easier ground and automatically progress to harder terrains as the robot succeeds (the code uses `_update_terrain_curriculum()` to adjust terrain level per environment). This ensures the robot first masters flat ground, then small obstacles, then larger ones.
* **Friction and Restitution:** Continue using a wide friction range and restitution variations to simulate slippery or bouncy ground. The current ranges (`friction_range: [0.2, 1.3]`, `restitution_range: [0.0, 0.4]` in config) are appropriate. These will randomize ground friction from very slick (0.2) to very sticky (1.3) and add some elasticity in collisions, forcing the policy to generalize its foot placement and balance.

## Control & Perception Noise (Delays and Sensor Jitter)

To make the policy robust to sensor latency and actuation errors, use the built-in lag and noise parameters:

* **Observation Lag (Sensor Delay):** The project already supports adding latency to observations. In config, enable IMU and DOF lag as needed. For example, set `add_imu_lag: True` and specify `imu_lag_timesteps_range: [1, 10]` to randomize a delay of 1–10 simulation steps in base orientation/angular velocity sensing. Similarly, `add_dof_lag: True` with `dof_lag_timesteps_range: [0, 40]` is already enabled, meaning joint sensor readings can be up to 40 steps out-of-date. These settings intentionally introduce observation delays so the policy learns to cope with stale information.
* **Actuation Delay (Control Latency):** You can simulate control command delay by holding older actions. In config, `add_dof_pos_vel_lag` can be used if you separate position/velocity command delays (the current snippet shows it off, but `add_dof_lag` covers a general delay). Another approach: implement a small action buffer in code. For instance, in `LeggedRobot.step()`, apply the previous action instead of the current one for a fixed number of sub-steps to simulate a latency of, say, 50 ms. This can be done by queuing actions and popping the delayed one each step. The provided config’s `lag_timesteps_range: [5, 40]` under `domain_rand` suggests a random 5–40 step delay might already be considered for actions as well. Ensure these flags are True to activate the delays.
* **Sensor Noise:** Continue to include random noise in observations (`add_noise: True`). The noise scales (e.g. slight noise on DOF positions, velocities, IMU quaternion, etc. already set in config) will simulate sensor jitter. These are currently scaled by `noise_level = 1.5` (50% higher than baseline). This level is quite significant; you can tune it up or down to balance realism and learning stability. Additionally, consider randomizing IMU bias or adding gaussian drift to sensors if needed (not in default code, but could be added similarly to other noise scales).

By enabling the above, each training episode will randomize delays and noises so that the policy experiences observation out-of-sync scenarios and must rely on a feedback mechanism (e.g. an LSTM or the history of states) to remain stable.

## Fast Recovery as a Reward Objective

To explicitly encourage the robot to regain balance **as quickly as possible after a disturbance**, modify the reward function to penalize prolonged instability. Currently, the orientation reward gives a smooth bonus for staying upright, but we can add a sharper incentive for *quick recovery*:

* **Balance Recovery Reward:** Introduce a new reward term that is **1 when the robot is upright and stable, and 0 otherwise** (or conversely a penalty when unstable). For example, define *“balanced posture”* as having the torso within a small angle from vertical. In code, we can check the robot’s roll and pitch (e.g. using `self.base_euler_xyz`). If |roll| and |pitch| are below a threshold (e.g. 0.3 rad ≈ 17°), consider the robot “balanced.” Each time step the robot meets this criterion, give a reward; if it exceeds it (significant tilt), give no reward (or a penalty). This creates pressure to minimize the time spent in a tilted state.

  * **Config:** Add a new reward scale in `X1DHStandCfg.rewards.scales`. For example, in **`humanoid/envs/x1/x1_dh_stand_config.py`**, insert:

    ```python
    class rewards:
        ...
        class scales:
            ...
            orientation = 1.0
            feet_rotation = 0.3
            base_height = 0.2
            base_acc = 0.2
            **balance_recovery = 1.0**    # new reward for staying balanced (quick recovery)
            action_smoothness = -0.002
            torques = -8e-9
            ...
    ```

    Place this around the existing reward scales (e.g. after `base_acc` and before the energy or limits terms). A scale of 1.0 is a starting point – it means up to 1 reward per time-step when balanced. You may adjust this weight (e.g. 0.5 or 2.0) to balance it against other rewards.
  * **Code:** Implement the corresponding reward function in **`tasks/locomotion.py`** (for X1, this is the env class `X1DHStandEnv`). For instance, after the `_reward_orientation` function, add:

    ```python
    def _reward_balance_recovery(self):
        """Reward for maintaining upright balance (quick post-disturbance recovery)."""
        # Calculate absolute roll and pitch angles from base orientation
        roll = torch.abs(self.base_euler_xyz[:, 0])
        pitch = torch.abs(self.base_euler_xyz[:, 1])
        # Define stability threshold (e.g. 0.3 rad ~ 17 degrees)
        stable = (roll < 0.3) & (pitch < 0.3)
        # Reward 1.0 if stable, 0.0 if not
        return stable.float()
    ```

    This will output **1** for each environment that is within the tilt limit, and 0 for those tilted beyond 0.3 rad. With the `balance_recovery` scale set positive (e.g. +1.0), the robot earns extra reward for every step it remains (or returns) upright. Essentially, after a push knocks it, it stops earning this reward until it regains upright posture – incentivizing it to recover faster. If you prefer a penalty formulation, you could have it return `unstable = (roll >= 0.3 | pitch >= 0.3).float()` and set the scale to a negative value (to directly penalize being off-balance), but using a positive reward for stability works well and avoids issues with total negative rewards (since `only_positive_rewards=True` in config).
  * **Integration:** Make sure this new reward is included in the total. The Legged Gym framework will automatically include any `_reward_*` function that has a corresponding non-zero scale in the config. By adding `balance_recovery` to scales and defining `_reward_balance_recovery`, it will be invoked each step. This reward explicitly encodes *“minimal recovery time”* – every timestep of delay in regaining balance is a missed opportunity to get +1, so the agent is motivated to stabilize in fewer timesteps.
  * **Reference:** The existing orientation reward uses a smooth exponential of tilt (which gives partial credit even for moderate tilts). Our added term is a stricter binary reward that kicks in only when near upright, complementing the orientation term. Together, they ensure the robot not only *tries* to stay upright (orientation reward) but also *urgently corrects* any large deviation (balance\_recovery reward).

* **Velocity Recovery:** In addition to posture, encourage the robot to resume its target gait velocity quickly. The current reward already measures velocity tracking error (`tracking_lin_vel`, `tracking_ang_vel`) using an exponential penalty. To emphasize quick recovery, you can increase the weight of these tracking terms or introduce a short-term penalty for large deviations:

  * For example, increase `tracking_lin_vel` scale from 1.8 to, say, **2.5**, and `tracking_ang_vel` from 1.1 to **1.5** in the config. This makes velocity mismatches (like being shoved off the commanded speed) more costly, so the agent will correct its velocity sooner.
  * You might also add a term that specifically punishes *prolonged* velocity error. One approach is a **“low-speed penalty”**: the code already contains `_reward_low_speed` which checks if the robot is moving too slow/fast relative to command and even penalizes opposite-direction movement strongly. Ensuring this is enabled (it appears in scales as `low_speed = 0.2`) will penalize failing to regain commanded speed (for instance, after a push when commanded to walk forward, stalling or moving backward yields a negative reward). You can consider increasing this weight or adjusting its thresholds (e.g. require speed to come back above 50% of commanded within a time window).
  * Combined with the push and orientation rewards, the velocity tracking rewards naturally drive quick gait recovery. After a push, the robot that quickly resumes the commanded forward velocity and correct heading will earn higher tracking rewards (exponential tracking error decays as error shortens), whereas a slow recovery means many timesteps of tracking error (near-zero reward each step). By tuning the scales upward, we make this effect stronger.

* **Footing and Stumble Recovery:** Encourage reactive steps to recover balance. You already have terms like `feet_air_time` (reward for consistent gait timing) and `foot_slip` (penalty for dragging feet). To directly reward *taking a quick catch-step*, you could monitor contact timing: for example, if a push occurs and the swing foot promptly contacts the ground (shortening the gait cycle to arrest a fall), that could be rewarded. Implementing this precisely is complex, but you can approximate it by penalizing **foot contact mismatches**:

  * The code uses `feet_contact_number` and `stance_mask` to compare expected vs actual contacts. Make sure the `feet_contact_number` reward (scale 2.0 in config) is active – this likely rewards having the appropriate number of feet on ground (e.g. both feet down when standing, one foot up when stepping). By maintaining this reward, the robot is encouraged to **plant its foot quickly if it’s supposed to** (for stability). If a push knocks it such that both feet leave the ground or an extra footfall is needed, it will lose this reward until it gets the contact pattern back to normal.
  * You might also slightly increase `foot_slip` penalty (currently -0.1) to discourage sliding after a push – forcing the robot to *lift and place* feet rather than skid.
  * **Contact state awareness:** as described next, adding contact info to observations will help the policy learn when a foot is on ground or not, which can improve its ability to time recovery steps.

By refining the reward function with these additions, **“time to recover to a stable gait”** becomes an explicit part of the optimization: every missed step of upright posture or correct velocity costs reward, so the agent will learn to recover faster to maximize returns.

## Observation Space Enhancements

It can be beneficial to provide the policy with additional sensory cues about disturbances and contacts, so it can react faster:

* **External Push Indicator:** Consider adding a feature to the observation that signals when a push occurs. For instance, a binary **“push\_flag”** that is 1 on the timestep of a push impulse and 0 otherwise can alert the policy to unexpected momentum changes. In a real robot this might correspond to a sudden spike in accelerometer readings – here we simplify it as a direct flag. Implementation:

  * In the environment code (`compute_observations` in `X1DHStandEnv`), after computing the usual obs components, append a feature for the push. The simulator knows when it applied a push – specifically, when `self.common_step_counter % push_interval == 0` right after incrementing. We can exploit that:

    ```python
    push_flag = (self.cfg.domain_rand.push_robots and 
                 (self.common_step_counter % self.cfg.domain_rand.push_interval) == 0)
    push_flag = push_flag.float().unsqueeze(1)  # shape (num_envs,1)
    ```

    Then include `push_flag` in the `obs_buf`. For example, in **`compute_observations`**, where the obs tensor is constructed, append the flag:

    ```python
    obs_buf = torch.cat((
        ...  # existing state features
        self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
        self.lagged_base_euler_xyz * self.obs_scales.quat,   # 3
        push_flag,                                          # 1 (new)
        contact_mask.float()                                # 2 (new)
    ), dim=-1)
    ```

    Make sure to update `num_single_obs` accordingly (adding 3 new features in this example) in the config. For instance, if it was 47, increase it to **50**. The code already calculates `num_observations = frame_stack * num_single_obs`, so the total observation dimension will adjust automatically (e.g. 66 \* 50 = 3300).

    > **Note:** The AgiBot X1 training code uses a history of 66 frames per observation, often processed by a CNN/LSTM, so a one-step push flag will appear as a brief spike in that sequence. The policy can learn to recognize this spike pattern (or the corresponding IMU jolt) and respond appropriately (e.g. shifting stance quickly).
* **Foot Contact State:** Providing the policy with which feet are in contact can improve balance control. Add two observation entries for left and right foot contact. The env already computes a boolean `contact_mask` for the feet (true if foot force > 5N). We saw that **privileged** observations (used by the critic) already include `stance_mask` and `contact_mask`. We should feed similar info to the policy (actor) as well:

  * Use the `contact_mask` (size 2) and append it to `obs_buf` (as shown in the code snippet above). Casting to float (1.0 or 0.0) is fine. This explicitly tells the policy which foot is on the ground at the current time.
  * With contact data, the robot can reason about its support phase. For instance, if a push occurs and one foot is in swing (contact=0), the policy might quickly expedite landing that foot (since it **knows** it's currently airborne). Without this info, it would have to infer foot contact indirectly from joint data or phase.
  * Update `num_single_obs` to account for the two contacts (already included in the +3 adjustment above).
* **Disturbance Measurements:** In lieu of a simple flag, one could also feed a richer measurement of the push – e.g. the external force vector or base acceleration. In the current code, the privileged critic gets `rand_push_force` and `rand_push_torque` values injected into its observations. To keep things simple for the policy, the binary flag plus the robot’s own IMU readings (linear/angular velocity) are usually sufficient. If you find the policy still struggles, you could provide the *magnitude and direction* of the push impulse (e.g. 2 numbers for force XY direction) to the actor as well. This is less realistic but gives the policy exact knowledge of the perturbation.
* **Verify Observation Lengths:** After adding new obs features, double-check that the config’s `num_single_obs` matches the new length. For example, if we added 3 features (push\_flag + 2 contacts) to the original 47, set:

  ```python
  num_single_obs = 50
  num_observations = int(frame_stack * num_single_obs)  # becomes 3300 with frame_stack=66
  ```

  (This would be in `X1DHStandCfg.env` around line 42-45.) Also adjust `single_num_privileged_obs` if needed for any new privileged signals (though in this case, privileged already had those signals; we mainly added to actor).
* **Curriculum Signals:** If using curriculum, you might also feed the current curriculum level or phase to the policy (though not strictly necessary). For example, if push magnitude increases in stages, the policy could benefit from knowing the current stage. This could be as simple as an integer stage index or the current `push_duration` value as part of observations. However, since the policy should ideally handle any magnitude, this isn’t essential.

With these observation tweaks, the **policy will be more aware of disturbances and its own contact situation**, allowing it to respond reflexively. This can shorten the recovery time by a few critical timesteps.

## Curriculum and Stage Training Strategy

Finally, employ a **staged training approach** to progressively build the robot’s robustness:

* **Stage 1: Baseline Gait Stability.** Train the policy in a relatively benign environment first. For example, disable pushes and use mostly flat terrain initially. You could set `push_robots = False` (or push interval very high) and maybe reduce noise for this phase. The goal is for the robot to learn a stable walking gait or standing balance *without* heavy disturbances.
* **Stage 2: Introduce Disturbances.** Once the baseline policy is converged, start a new training (or continue training) with disturbances enabled. Turn on `push_robots` (with the configured frequency/magnitude), uneven terrain, and noise/delay as per the earlier settings. Because the policy starts from a competent gait (by loading the Stage 1 model), it will adapt to recover from pushes much faster than if it had to learn locomotion and disturbance rejection from scratch.
* **Gradual Domain Randomization:** Within Stage 2, apply curriculum **during training** as well:

  * We already discussed gradually increasing push intensity. The config example with `push_duration` list is a practical implementation: the first few thousand iterations have no pushes, then small pushes, and so on. Make sure the environment code uses `push_duration` appropriately (e.g. applying an impulse over multiple simulation steps if duration > 0). If that logic isn’t implemented, an alternative is to start with `max_push_vel_xy` very low and manually raise it after certain checkpoints.
  * Similarly, you can widen the terrain difficulty over time. If `terrain.curriculum=True`, the built-in terrain manager will already move bots that perform well to harder sections. If not using that, you can manually ramp up terrain proportions (e.g. increase the probability of stairs or deep gaps every few million steps).
  * **Noise/Delay Curriculum:** Start with modest noise and shorter delays, then increase. For instance, begin with `imu_lag_timesteps_range = [1,5]` and later expand to \[1,10]. Or set `noise_level = 1.0` initially, then 1.5 or 2.0 as training progresses.
  * In code, one way to do this is to tie these parameters to the training iteration count. Since the config is mostly static at start, you might implement a check in the training loop to adjust env parameters at certain iteration milestones. For example, after 500 iterations, set `env.cfg.domain_rand.imu_lag_timesteps_range = [1, 10]`. This could be done via a callback or by reinitializing environments with updated config (depending on how the training script is structured). Another method is as shown with `update_step` in config, where the environment itself checks `common_step_counter` or similar to transition its settings.
* **Evaluation of Recovery Time:** As you train Stage 2, monitor the *“disturbance recovery time”* metric to see progress. You can log how many seconds (or simulation steps) after a push it takes for certain conditions to be met (e.g. tilt < 5° and velocity error < some threshold). This isn’t used in reward directly (since we approximated it with per-step penalties), but it’s a useful evaluation metric to ensure the training is achieving the goal. Over the training run, this average recovery time should decrease.

By combining a careful curriculum with the above reward and observation modifications, the robot will **first learn to walk**, then **learn to withstand pushes**, and finally **learn to recover its gait rapidly** after those pushes. This staged approach ensures stability is not sacrificed: the policy won’t get “stuck” learning to recover if it hasn’t learned to walk yet, and once it can handle one disturbance, we incrementally challenge it more.

---

**Summary of Key Code Modifications:**

* **Environment Config (`*.yaml` or Python config):** Enable and tune disturbance parameters:

  * Set push frequency and magnitude (e.g. 4s interval, moderate initial velocity).
  * Expand terrain types (enable slopes, stairs, etc.) and use friction/randomization ranges.
  * Turn on sensor/actuator delay models (`add_dof_lag`, `add_imu_lag`, etc.) with appropriate ranges.
  * Add curriculum parameters (like `push_duration` schedule, `curriculum=True` for terrain and commands) to gradually intensify disturbances.
* **Reward Function (`tasks/locomotion.py` or env class):** Add a **balance recovery reward**. Insert a new scale (e.g. `balance_recovery`) in the reward config and implement `_reward_balance_recovery()` in the env class to reward timely restoration of upright posture. Also consider boosting existing reward scales for orientation and velocity tracking to weight quick recovery more heavily.
* **Observation Space (env `compute_observations`):** Append **push indicators and contact state** to the policy observations. Increase `num_single_obs` (e.g. from 47→50). In code, include a `push_flag` and foot contact info when constructing `obs_buf`. For example, after base velocities and angles, add:

  ```python
  obs_buf = torch.cat((..., self.lagged_base_euler_xyz*scale, push_flag, contact_mask.float()), dim=-1):contentReference[oaicite:62]{index=62}.
  ```

  This gives the policy immediate knowledge of external perturbations and its support footing.
* **Curriculum Logic:** Either use config schedules (like the provided push\_duration list and `update_step` logic) or implement a manual training loop adjustment to **phase in disturbances**. For instance, start with `push_robots=False`; then after N iterations or upon loading the next stage, set `push_robots=True` and update the config for greater randomization (you might script this by saving a checkpoint at stage 1 completion and then editing the config for stage 2 before resuming training).

With these changes, your AgiBot X1 training should produce a policy that not only maintains **robust balance under random pushes, uneven terrain, and sensor noise**, but also **recovers its stable gait within minimal time** after a disturbance. The reward modifications directly target rapid recovery, and the variety introduced through domain randomization and curriculum will ensure the robot can handle a wide range of real-world perturbations. Good luck with training **v1.6 (disturbance resilience)**! 🚀

**Sources:**

* AgiBot X1 Training Config (domain randomization & push settings)
* AgiBot X1 Reward Function Snippets (orientation reward, scales definition)
* Observation construction and privileged info (for adding push/contact signals)
* Legged Gym Reference (random push and terrain curriculum mechanics)

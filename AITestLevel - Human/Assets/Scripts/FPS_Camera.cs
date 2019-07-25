using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System;
using System.Threading;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEditor;
using System.Net;
using System.Timers;

// main script for player
public class FPS_Camera : MonoBehaviour
{
    public GameObject Player;
    public Camera PlayerCamera;
    public Animator Anim;
    public GameObject Door;
    public Text tex;
    public int reward;
    public int totalReward;
    public bool DQN;

    private CharacterController RbPlayer;
    private Collider DoorCollider;
    private Transform RbTransform;
    private Transform CameraTransform;
    private Vector3 moveDirection = Vector3.zero;
    private Vector3 startingPos;
    private Quaternion startingRot;
    private Socket clientPass;
    private IPAddress ipAd;
    private string action;
    private int param;
    private float xmovement;
    private float ymovement;
    private static System.Timers.Timer aTimer;
    private static System.Timers.Timer bTimer;
    private bool move;
    private bool nextEp;
    private GameObject[] bulletArrayy;


    // The port number for the remote device.  
    private const int port = 12345;

    // ManualResetEvent instances signal completion.  
    private Socket _clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
    private byte[] _recieveBuffer = new byte[1024];

    // Start is called before the first frame update
    void Start()
    {
      
        WriteString("NEW SESSION" + System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss"));
        
        aTimer = new System.Timers.Timer();
        aTimer.Elapsed += new ElapsedEventHandler(OnTimedEvent);
        aTimer.Interval = 120000;
        aTimer.Enabled = true;

        bTimer = new System.Timers.Timer();
        bTimer.Elapsed += new ElapsedEventHandler(OnTimedEvent2);
        bTimer.Interval = 10000;
        bTimer.Enabled = true;

        if (DQN == true)
        {
            ipAd = GetLocalIPAddress();
            StartClient();
        }

        
        RbPlayer = Player.GetComponent<CharacterController>();
        RbTransform = RbPlayer.GetComponent<Transform>();
        CameraTransform = PlayerCamera.GetComponent<Transform>();
        DoorCollider = Door.GetComponent<Collider>();
        RbPlayer.detectCollisions = true;
        move = true;
        nextEp = false;
        reward = 0;
        xmovement = 0;
        ymovement = 0;
        startingPos = RbTransform.position;
        startingRot = RbTransform.rotation;


    }

    // Update is called once per frame
    void FixedUpdate()
    {
        //connect to dqn
        if (DQN == true)
        {
            _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
            StartCoroutine(HandleInput());
        }
       

        if (Input.GetKey("right"))
        {
            StartCoroutine(Right());
        }

        if (Input.GetKey("left"))
        {
            StartCoroutine(Left());
        }

        if (RbPlayer.isGrounded)
        {
            if (Input.GetKey("up"))
            {
                StartCoroutine(Up());
            }

            if (Input.GetKey("down"))
            {
                StartCoroutine(Down());
            }
            moveDirection = Vector3.zero;
        }

        if (Input.GetKeyDown("space"))
        {
            Debug.Log("Space");
            Debug.Log(reward);
            StartCoroutine(fireBullet());
        }

        //gravity
        moveDirection.y = moveDirection.y - (10.0f * Time.deltaTime);
        // Move the controller
        RbPlayer.Move(moveDirection * Time.deltaTime);
        xmovement = RbTransform.position.x - startingPos.x;
        ymovement = RbTransform.position.y - startingPos.y;
        bugs();
    }

    //handle collisions
    void OnControllerColliderHit(ControllerColliderHit col)
    {
        if (col.gameObject.CompareTag("Fire"))
        {
            string action = System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "Fire Check Point";
            WriteString(action);
            StartCoroutine(GameEnd());
            cleanUpBullets();
            Debug.Log("Restart");
        }
        if (col.gameObject.CompareTag("chair"))
        {
            move = false;
            string bug = System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "chair bug at " + RbPlayer.transform.position;
            WriteString(bug);
            move = true;
            StartCoroutine(GameEnd());
            cleanUpBullets();
        }
        if (col.gameObject.CompareTag("stairs"))
        {
            string action = System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "Stairs Check Point";
            WriteString(action);
            reward += 101;
        }
        if (col.gameObject.CompareTag("room"))
        {
            string action = System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "Hidden room Check Point";
            WriteString(action);
            
        }
        
    }
    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("incent"))
        {
            reward += 10;
        }
    }
    //Handle input
    private IEnumerator HandleInput()
    {
        
        switch (action)
        {
            case "Up":
                StartCoroutine(Up());
                action = "";
                break;
            case "Down":
                StartCoroutine(Down());
                action = "";
                break;
            case "Left":
                StartCoroutine(Left());
                action = "";
                break;
            case "Right":
                StartCoroutine(Right());
                action = "";
                break;
            case "fireBullet":
                StartCoroutine(fireBullet());
                action = "";
                break;
            case "ScreenCap":
                StartCoroutine(ScreenCap());
                action = "";
                break;
            case "GameEnd":
                StartCoroutine(GameEnd());
                cleanUpBullets();
                action = "";
                break;
            case "SendData":
                SendData(param);
                action = "";
                break;
        }
        yield return null;
        }
        private IEnumerator fireBullet()
    {
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.tag = "Bullet";
        sphere.AddComponent<Rigidbody>();
        Rigidbody sphereRB = sphere.GetComponent<Rigidbody>();
        sphereRB.useGravity = false;
        sphere.transform.localScale += new Vector3(-0.5f, -0.5f, -0.5f);
        sphere.transform.position = CameraTransform.position + Camera.main.transform.forward * 100.0f * Time.deltaTime;
        sphere.transform.rotation = CameraTransform.rotation;
        sphereRB.AddForce(new Vector3(sphere.transform.position.x - CameraTransform.position.x, 0, sphere.transform.position.z - CameraTransform.position.z) * 600.0f);

        yield return null;
    }

      public void WriteString(string bug)
    {
        string path = Application.dataPath + "/Test.txt";

        //Write some text to the test.txt file
        StreamWriter writer = new StreamWriter(path, true);
        writer.WriteLine(bug);
        writer.Close();
    }

    
    private IEnumerator Up()
    {
        if (move == true)
        {
            moveDirection = transform.TransformDirection(Vector3.forward);
            moveDirection = moveDirection * 10.0f;
            RbPlayer.Move(moveDirection * Time.deltaTime);
            if (xmovement % 2 == 0)
            {
                reward += 1;
            }
            if (ymovement % 2 == 0)
            {
                reward += 1;
            }
            yield return null;
        }
    }

    private IEnumerator Down()
    {
        if (move == true)
        {
            moveDirection = transform.TransformDirection(Vector3.back);
            moveDirection = moveDirection * 10.0f;
            RbPlayer.Move(moveDirection * Time.deltaTime);
            if (xmovement % 2 == 0)
            {
                reward += 1;
            }
            if (ymovement % 2 == 0)
            {
                reward += 1;
            }
            yield return null;
        }
    }

    private IEnumerator Left()
    {
        if (move == true)
        {
            RbTransform.RotateAround(RbTransform.position, RbTransform.up, Time.deltaTime * -100f);
            yield return null;
        }
    }

    private IEnumerator Right()
    {
        if (move == true)
        {
            RbTransform.RotateAround(RbTransform.position, RbTransform.up, Time.deltaTime * 100f);
            yield return null;
        }
    }

    

    private IEnumerator GameEnd()
    {
        RbTransform.position = startingPos;
        RbTransform.rotation = startingRot;
        //SendData(-999);
        totalReward = 0;
        nextEp = true;
        move = true;
        
       
        yield return null;
    }

    public static IPAddress GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                return ip;
            }
        }
        throw new Exception("No network adapters with an IPv4 address in the system!");
    }

    private void StartClient()
    {

        try
        {
            _clientSocket.BeginConnect(new IPEndPoint(ipAd, 12345), new AsyncCallback(ConnectCallback), null);
        }
        catch (SocketException ex)
        {
            Debug.Log(ex.Message);

        }
        Debug.Log("connected");

        _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);

    }

    private void ConnectCallback(IAsyncResult AR)
    {
        try
        {
            // Complete the connection.  
            _clientSocket.EndConnect(AR);
        }
        catch (Exception e)
        {
            Console.WriteLine(e.ToString());
        }
    }
    private void ReceiveCallback(IAsyncResult AR)
    {
        //Check how much bytes are recieved and call EndRecieve to finalize handshake
        int recieved = _clientSocket.EndReceive(AR);
        
        if (recieved <= 0)
            return;
        
        //Copy the recieved data into new buffer , to avoid null bytes
        byte[] recData = new byte[recieved];
        Buffer.BlockCopy(_recieveBuffer, 0, recData, 0, recieved);

        
        //Processing received data
        Debug.Log(System.Text.Encoding.ASCII.GetString(recData));
        string serverMessage = Encoding.ASCII.GetString(recData);
        
        Array.Clear(recData, 0, recData.Length);
        

        if (serverMessage == "4")
        {
            action = "Up";

        }
        if (serverMessage == "2")
        {
            action = "Down";
        }
        if (serverMessage == "1")
        {
            action = "Left";
        }
        if (serverMessage == "5")
        {
            action = "Right";
        }
        if (serverMessage == "3")
        {
            action = "fireBullet";
        }
        if (serverMessage == "S")
        {
            action = "ScreenCap";
        }
        if (serverMessage == "N")
        {
            nextEp = false;
        }
        if (serverMessage == "R")
        {
            param = reward;
            action = "SendData";
            totalReward += reward;
            reward = 0;
        }
        if (serverMessage == "T")
        {
            totalReward = 0;
        }



    }

   private IEnumerator ScreenCap()
    {
        ScreenCapture.CaptureScreenshot("SomeLevel.PNG");
        yield return null;
    }

    private void SendData(int data)
    {   
        if(nextEp == true)
        {
            data = -999;
        }
        string clientMessage = data.ToString();
        // Convert string message to byte array.                 
        byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(clientMessage);

        SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
        socketAsyncData.SetBuffer(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);
        _clientSocket.BeginSend(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length, 0,
           new AsyncCallback(SendCallback), null);
        //_clientSocket.Send(clientMessageAsByteArray, SocketFlags.None);

    }

    private void SendCallback(IAsyncResult ar)
    {
        try
        {
            // Retrieve the socket from the state object.  
            // Complete sending the data to the remote device.  
            int bytesSent = _clientSocket.EndSend(ar);
        }
        catch (Exception e)
        {
            Console.WriteLine(e.ToString());
        }
    }

    private void OnTimedEvent(object source, ElapsedEventArgs e)
    {
        action = "GameEnd";
    }

    private void OnTimedEvent2(object source, ElapsedEventArgs e)
    {
        SendData(reward);
    }


    private void bugs()
    {
        if (RbPlayer.transform.position.y < -50)
        {
            string bug = "falling bug at " + RbPlayer.transform.position;
            WriteString(System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + bug);
            StartCoroutine(GameEnd());
            cleanUpBullets();
        }

        bulletArrayy = GameObject.FindGameObjectsWithTag("Bullet");
        if (bulletArrayy.Length > 5)
        {
            WriteString(System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "visable projectiles bug at " + RbPlayer.transform.position);
        }
    }

    private void cleanUpBullets()
    {
        foreach (GameObject bullet in bulletArrayy)
        {
            Debug.Log("bullet found");
            Destroy(bullet);

        }
    }
    IEnumerator Wait()
    {
       
        yield return new WaitForSeconds(4);
       
    }

}

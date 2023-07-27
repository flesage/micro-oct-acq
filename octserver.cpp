#include <iostream>
#include <QtWidgets>
#include <QtNetwork>
#include <QDir>
#include "config.h"

#include "octserver.h"

// TODO: Refactor to remove the fft reconstruction from the server, send the f_fft as input
// TODO: Set the server in overwrite mode.
// TODO: add method to reset scan parameters and fft
// TODO: move the remote saver to a separate class

OCTServer::OCTServer(QWidget *parent, int nx, int n_extra, int n_alines, int n_repeat, int factor)
    : QDialog(parent)
    , statusLabel(new QLabel)
    , tcpServer(new QTcpServer),
      f_fft(n_repeat, factor) // TODO: use n_repeat and factor for the f_fft constructor
{
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
    p_request_type = REQUEST_TILE;
    p_nx = nx;
    p_n_extra = n_extra;
    p_n_alines = n_alines;
    p_put_done = 0;
    p_factor = factor;
    p_current_pos = 0;
    p_frame_size = (p_nx + p_n_extra)*LINE_ARRAY_SIZE * p_factor;
    p_buffer_size = p_frame_size * 2;


    initServer();
    auto quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);

    // Prepare the data reconstruction
    float dimz = 3.5; // FIXME: dummy axial resolution
    float dimx = 3.0; // FIXME: dummy lateral resolution
    f_fft.init(LINE_ARRAY_SIZE, factor*(p_nx + p_n_extra), dimz, dimx);
    p_fringe_buffer = new unsigned short[p_buffer_size];
    p_image_buffer = new float[p_frame_size/2];

    // Connections
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::close);
    connect(tcpServer, &QTcpServer::newConnection, this, &OCTServer::slot_startConnection);

    // Layout
    auto buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    QVBoxLayout *mainLayout = nullptr;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        auto outerVerticalLayout = new QVBoxLayout(this);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
        auto outerHorizontalLayout = new QHBoxLayout;
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        auto groupBox = new QGroupBox(QGuiApplication::applicationDisplayName());
        mainLayout = new QVBoxLayout(groupBox);
        outerHorizontalLayout->addWidget(groupBox);
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        outerVerticalLayout->addLayout(outerHorizontalLayout);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
    } else {
        mainLayout = new QVBoxLayout(this);
    }

    mainLayout->addWidget(statusLabel);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
}

OCTServer::~OCTServer()
{
    tcpServer->close();
}

void OCTServer::initServer()
{
    // Set the server address and port
    QString ipAddress = OCTServer::getHostAddress();
    const QHostAddress address = QHostAddress(ipAddress);
    quint16 ipPort = SERVER_SOCKET_PORT;

    if (!tcpServer->listen(address, ipPort)) {
        QMessageBox::critical(this, tr("OCT Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2\n\n"
                            "Waiting for connections ...")
                         .arg(ipAddress).arg(tcpServer->serverPort()));

}

void OCTServer::put(unsigned short* fringe)
{
    memcpy(&p_fringe_buffer[(p_current_pos % p_buffer_size)*p_frame_size], fringe, p_frame_size*sizeof(unsigned short));
    p_current_pos++;
}

void OCTServer::slot_endConnection()
{
    std::cerr << "octserver::Closing the connection ";
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    int n_bytes;
    out.setVersion(QDataStream::Qt_6_4);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);
    switch (p_request_type) {
    case REQUEST_TILE:
        // TODO: send the number of bytes
        n_bytes = 9 * sizeof(char) + 2 * sizeof(int); // QDataStream magic number + n_bytes + Message
        out << n_bytes;
        out << "OCT_done";
        break;
    case REQUEST_BSCAN_NOSAVE:
        f_fft.image_reconstruction(p_fringe_buffer, p_image_buffer, 1, 1024);

        int n_x = p_nx + p_n_extra;
        int n_y = p_factor;
        int n_z = (LINE_ARRAY_SIZE / 2);
        n_bytes = n_x * n_y * n_z * sizeof(float) + 4 * sizeof(int);

        // Add the data to the byte array
        out << n_bytes;
        out << n_x;
        out << n_y;
        out << n_z;

        for (int i=0; i< n_x * n_y * n_z; i++) {
           out << p_image_buffer[i];
        }
        p_current_pos = 0;
        break;
    }
    std::cerr << "(transmitting " << n_bytes << " bytes)... ";
    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
    p_mutex.lock();
    p_put_done = 0;
    p_mutex.unlock();
}

QString OCTServer::getHostAddress()
{
    QString ipAddress;
    const QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();

    // use the first non-localhost IPv4 address
    for (const QHostAddress &entry : ipAddressesList) {
        if (entry != QHostAddress::LocalHost && entry.toIPv4Address()) {
            ipAddress = entry.toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return ipAddress;
}

void OCTServer::slot_parseRequest()
{
    QByteArray total_data, buffer;
    while(1) {
        buffer = clientConnection->read(1024);
        if (buffer.isEmpty()) {
            break;
        }
        total_data.append(buffer);
    }
    QString data = QString(total_data);

   if ( data.startsWith("request_autofocus_bscan"))
   {
       std::cerr << "Case 1: Autofocus B-Scan Request" << std::endl;
       p_request_type = REQUEST_BSCAN_NOSAVE;
       emit sig_set_request_type(QString("autofocus"));
       setup_and_request_scan();
   }
   else
   {
       std::cerr << "Case 2: Tile Request" << std::endl;
       p_request_type = REQUEST_TILE;
       emit sig_set_request_type(QString("tile"));
       std::cerr << "octserver::Reading tile positions: (x,y,z)=";
       // Read the tile position to acquire
       p_tile_x = data.split(" ")[0].toInt();
       p_tile_y = data.split(" ")[1].toInt();
       p_tile_z = data.split(" ")[2].toInt();

       std::cerr << "(" << p_tile_x <<"," << p_tile_y << "," << p_tile_z << ")" << std::endl;
        setup_and_request_scan(p_tile_x, p_tile_y, p_tile_z);
   }
}


void OCTServer::slot_startConnection()
{
    std::cerr << "octserver::Fetching the next pending connection...";
    clientConnection = tcpServer->nextPendingConnection();
    std::cerr << "got it!" << std::endl;

    connect(clientConnection, &QIODevice::readyRead, this, &OCTServer::slot_parseRequest);
}

void OCTServer::setup_and_request_scan(int x, int y, int z)
{
    // Create the tile filename
    char buffer [20];
    snprintf(buffer, 20, "tile_x%02d_y%02d_z%02d", x, y, z);
    QString fileName = QString(buffer);
    emit sig_change_filename(fileName);

    // Launch a single acquisition
    emit sig_start_scan();
}

void OCTServer::setup_and_request_scan()
{
    // Create the tile filename
    QString fileName = QString("dummy");
    emit sig_change_filename(fileName);

    // Launch a single acquisition
    emit sig_start_scan();
}

